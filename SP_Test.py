import os
import json
import time
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import jsonschema
import google.generativeai as genai
from dotenv import load_dotenv
from sp_api.api import ProductTypeDefinitions, Feeds
from sp_api.base import Marketplaces

load_dotenv()


@dataclass
class Config:
    APP_ID: str = os.getenv("LWA_APP_ID_RELEASE", "")
    CLIENT_SECRET: str = os.getenv("LWA_CLIENT_SECRET_RELEASE", "")
    REFRESH_TOKEN: str = os.getenv("REFRESH_TOKEN_RELEASE", "")
    GEMINI_KEY: str = os.getenv("GEMINI_API_KEY", "")
    FILE_URL: str = 'https://docs.google.com/spreadsheets/d/1H1ii2VkycRlu52HdIoR1VU_iVn8IaVcEOyGhprdGDHc/export?format=csv&gid=0'
    MARKETPLACE: Marketplaces = Marketplaces.US
    DEFAULT_LANGUAGE: str = "en_US"
    FEED_TYPE: str = "JSON_LISTINGS_FEED"
    CONTENT_TYPE: str = "application/json; charset=UTF-8"
    FALLBACK_CANDIDATES: Tuple[str, ...] = ("MAJOR_HOME_APPLIANCES_PART", "HOME_APPLIANCE_ACCESSORY",
                                            "REPLACEMENT_PART")
    DRY_RUN: bool = True


class GeminiAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing Gemini API Key")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def identify_search_term(self, title: str) -> str:
        try:
            time.sleep(1.0)
            prompt = f"Analyze title: '{title}'. Identify Appliance Type. Use 'Washing Machine' not 'Washer'. Output ONLY Name."
            response = self.model.generate_content(prompt)
            return response.text.strip().replace('"', '').replace("'", "")
        except Exception:
            return "Appliance"

    def select_best_type(self, title: str, candidates: List[Dict[str, str]]) -> str:
        try:
            time.sleep(1.2)
            candidates_names = [c['name'] for c in candidates]
            prompt = f"Product: '{title}'\nCats: {json.dumps(candidates_names)}\nSelect best PART category. Output ONLY name."
            response = self.model.generate_content(prompt)
            selected = response.text.strip().replace('"', '').replace("'", "")
            if selected in candidates_names:
                return selected
            return candidates[0]['name']
        except Exception:
            if candidates:
                return candidates[0]['name']
            return "MAJOR_HOME_APPLIANCES_PART"


class AmazonClient:
    def __init__(self, config: Config):
        if not all([config.APP_ID, config.CLIENT_SECRET, config.REFRESH_TOKEN]):
            raise ValueError("Missing Amazon Keys")

        self.credentials = {
            'lwa_app_id': config.APP_ID,
            'lwa_client_secret': config.CLIENT_SECRET,
            'refresh_token': config.REFRESH_TOKEN,
        }
        self.marketplace = config.MARKETPLACE
        self.marketplace_id = [config.MARKETPLACE.marketplace_id]
        self.fallback_candidates = config.FALLBACK_CANDIDATES
        self.content_type = config.CONTENT_TYPE
        self.feed_type = config.FEED_TYPE

        self.definitions_client = ProductTypeDefinitions(
            marketplace=self.marketplace,
            credentials=self.credentials
        )
        self.feeds_client = Feeds(
            marketplace=self.marketplace,
            credentials=self.credentials
        )
        self.schema_cache = {}

    def search_product_types(self, query: str) -> List[Dict[str, str]]:
        candidates = []
        if query:
            try:
                response = self.definitions_client.search_definitions_product_types(
                    keywords=[query],
                    marketplaceIds=self.marketplace_id
                )
                if response.payload and 'productTypes' in response.payload:
                    for item in response.payload['productTypes']:
                        candidates.append({
                            'name': item['name'],
                            'displayName': item['displayName']
                        })
            except Exception:
                pass

        existing_names = {c['name'] for c in candidates}
        for fallback in self.fallback_candidates:
            if fallback not in existing_names:
                candidates.append({'name': fallback, 'displayName': fallback})

        return candidates

    def get_product_type_schema(self, product_type: str) -> Optional[Dict[str, Any]]:
        if product_type in self.schema_cache:
            return self.schema_cache[product_type]

        try:
            response = self.definitions_client.get_definitions_product_type(
                productType=product_type,
                marketplaceIds=self.marketplace_id,
                requirements="LISTING",
                locale="en_US"
            )
            schema = response.payload.get('schema')
            if schema:
                self.schema_cache[product_type] = schema
                return schema
        except Exception:
            pass
        return None

    def submit_feed(self, messages: List[Dict[str, Any]]) -> str:
        feed_body = {
            "header": {
                "sellerId": "SELLER_ID",
                "version": "2.0",
                "issueLocale": "en_US"
            },
            "messages": messages
        }

        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False, encoding='utf-8') as tmp:
                json.dump(feed_body, tmp)
                tmp_path = tmp.name

            doc_response = self.feeds_client.create_feed_document(
                file=tmp_path,
                content_type=self.content_type
            )
            feed_document_id = doc_response.payload.get('feedDocumentId')

            feed_response = self.feeds_client.create_feed(
                feedType=self.feed_type,
                inputFeedDocumentId=feed_document_id,
                marketplaceIds=self.marketplace_id
            )

            os.remove(tmp_path)
            return feed_response.payload.get('feedId')

        except Exception as e:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e


class PayloadBuilder:
    def __init__(self, schema: Dict[str, Any], config: Config):
        self.schema = schema
        self.language = config.DEFAULT_LANGUAGE

    def _clean_value(self, value: Any) -> Optional[str]:
        if pd.notna(value) and str(value).strip() and str(value).lower() != 'nan':
            return str(value).strip()
        return None

    def build_payload(self, product_type: str, row: pd.Series) -> Dict[str, Any]:
        attributes = {}

        title = self._clean_value(row.get('title'))
        if title:
            attributes['item_name'] = [{'value': title[:200], 'language_tag': self.language}]

        brand = self._clean_value(row.get('brand')) or 'Generic'
        attributes['brand'] = [{'value': brand, 'language_tag': self.language}]
        attributes['manufacturer'] = [{'value': brand, 'language_tag': self.language}]

        description = self._clean_value(row.get('product description'))
        if description:
            attributes['product_description'] = [{'value': description[:2000], 'language_tag': self.language}]

        bullets = []
        for i in range(1, 6):
            bullet = self._clean_value(row.get(f'bullet {i}'))
            if bullet:
                bullets.append({'value': bullet[:1000], 'language_tag': self.language})
        if bullets:
            attributes['bullet_point'] = bullets

        mpn = self._clean_value(row.get('MPN/Model Part Number'))
        if mpn:
            attributes['part_number'] = [{'value': mpn}]
            attributes['model_number'] = [{'value': mpn}]

        upc = self._clean_value(row.get('upc'))
        if upc:
            clean_upc = upc.replace('.0', '')
            if len(clean_upc) >= 11:
                attributes['externally_assigned_product_identifier'] = [{'type': 'upc', 'value': clean_upc}]

        image_url = self._clean_value(row.get('image1'))
        if image_url:
            attributes['main_product_image_locator'] = [{'media_location': image_url}]

        attributes['unit_count'] = [{'value': 1, 'type': 'count'}]
        attributes['number_of_items'] = [{'value': 1}]
        attributes['supplier_declared_dg_hz_regulation'] = [{'value': 'not_applicable'}]

        keyword = product_type.lower().replace('_', ' ').replace('part', 'parts')
        attributes['item_type_keyword'] = [{'value': keyword, 'language_tag': self.language}]

        return {
            "productType": product_type,
            "attributes": attributes
        }

    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        try:
            jsonschema.validate(instance=payload['attributes'], schema=self.schema)
            return True
        except jsonschema.ValidationError:
            return False


class InventoryProcessor:
    def __init__(self):
        self.config = Config()
        self.gemini = GeminiAnalyzer(self.config.GEMINI_KEY)
        self.amazon = AmazonClient(self.config)

    def process_file(self):
        try:
            df = pd.read_csv(self.config.FILE_URL)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        feed_messages = []

        for index, row in df.iterrows():
            sku = str(row.get('sku', ''))
            title = str(row.get('title', ''))

            if not title or title.lower() == 'nan':
                continue

            search_term = self.gemini.identify_search_term(title)
            candidates = self.amazon.search_product_types(search_term)

            if not candidates:
                print(f"SKU: {sku} - No candidates found")
                continue

            product_type = self.gemini.select_best_type(title, candidates)
            schema = self.amazon.get_product_type_schema(product_type)

            if not schema and candidates:
                for candidate in candidates:
                    if candidate['name'] != product_type:
                        fallback_schema = self.amazon.get_product_type_schema(candidate['name'])
                        if fallback_schema:
                            product_type = candidate['name']
                            schema = fallback_schema
                            break

            if not schema:
                print(f"SKU: {sku} - Schema not found for {product_type}")
                continue

            builder = PayloadBuilder(schema, self.config)
            payload = builder.build_payload(product_type, row)

            if builder.validate_payload(payload):
                print(f"SKU: {sku} - VALID - {product_type}")
                feed_messages.append({
                    "messageId": index + 1,
                    "sku": sku,
                    "operationType": "UPDATE",
                    "productType": product_type,
                    "attributes": payload["attributes"]
                })
            else:
                print(f"SKU: {sku} - INVALID - {product_type}")

        if feed_messages:
            if self.config.DRY_RUN:
                print(f"DRY RUN: Generated {len(feed_messages)} messages. No feed submitted.")
                with open("feed_output.json", "w") as f:
                    json.dump(feed_messages, f, indent=2)
            else:
                print(f"Submitting feed with {len(feed_messages)} messages...")
                try:
                    feed_id = self.amazon.submit_feed(feed_messages)
                    print(f"Feed submitted successfully. Feed ID: {feed_id}")
                except Exception as e:
                    print(f"Feed submission failed: {e}")
        else:
            print("No valid messages to submit.")


if __name__ == '__main__':
    processor = InventoryProcessor()
    processor.process_file()