import logging
import os
import json
import pandas as pd
from typing import Optional, Dict, List, Any, Set
from collections import defaultdict

logger = logging.getLogger("universal_interaction_tool")


class InteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir

        # Initialize data containers
        self.item_data: Dict[str, Dict] = {}
        self.user_data: Dict[str, Dict] = {}
        self.review_data: Dict[str, Dict] = {}

        # Initialize review indices
        self.item_reviews: Dict[str, List[Dict]] = defaultdict(list)
        self.user_reviews: Dict[str, List[Dict]] = defaultdict(list)

        # Load all data
        logger.info("Loading item data...")
        self._load_items()
        logger.info("Loading user data...")
        self._load_users()
        logger.info("Loading review data...")
        self._load_reviews()

    def _load_data(self, filename: str) -> List[Dict]:
        """Load data from a JSON Lines file."""
        file_path = os.path.join(self.data_dir, filename)
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
        return data

    def _load_items(self):
        """Load and process items from all sources."""
        items = self._load_data('item.json')
        for item in items:
            source = item.get('source', '')
            if source in ['yelp', 'amazon', 'goodreads']:
                item_id = item['item_id']
                self.item_data[item_id] = self._standardize_item(item)

    def _load_users(self):
        """Load and process users from all sources."""
        users = self._load_data('user.json')
        for user in users:
            source = user.get('source', '')
            if source in ['yelp', 'amazon', 'goodreads']:
                user_id = user['user_id']
                self.user_data[user_id] = self._standardize_user(user)

    def _load_reviews(self):
        """Load and process reviews from all sources."""
        reviews = self._load_data('review.json')
        for review in reviews:
            source = review.get('source', '')
            if source in ['yelp', 'amazon', 'goodreads']:
                review_id = review['review_id']
                standardized_review = self._standardize_review(review)
                self.review_data[review_id] = standardized_review

                # Build indices
                self.item_reviews[review['item_id']].append(standardized_review)
                self.user_reviews[review['user_id']].append(standardized_review)

    def _standardize_item(self, item: Dict) -> Dict:
        """Standardize item data based on source."""
        source = item.get('source', '')
        if source == 'yelp':
            return {
                'item_id': item['item_id'],
                'name': item.get('name', ''),
                'rating': item.get('stars', 0.0),
                'review_count': item.get('review_count', 0),
                'categories': item.get('categories', '').split(', '),
                'attributes': item.get('attributes', {}),
                'location': {
                    'address': item.get('address', ''),
                    'city': item.get('city', ''),
                    'state': item.get('state', ''),
                    'postal_code': item.get('postal_code', ''),
                    'latitude': item.get('latitude', 0.0),
                    'longitude': item.get('longitude', 0.0)
                },
                'is_open': item.get('is_open', 0),
                'hours': item.get('hours', {}),
                'source': 'yelp',
                'type': item.get('type', '')
            }
        elif source == 'amazon':
            return {
                'item_id': item['item_id'],
                'name': item.get('title', ''),
                'rating': item.get('average_rating', 0.0),
                'review_count': item.get('rating_number', 0),
                'categories': item.get('categories', []),
                'price': item.get('price', 0.0),
                'features': item.get('features', []),
                'description': item.get('description', []),
                'images': item.get('images', []),
                'details': item.get('details', {}),
                'store': item.get('store', ''),
                'source': 'amazon',
                'type': item.get('type', '')
            }
        elif source == 'goodreads':
            return {
                'item_id': item['item_id'],
                'name': item.get('title', ''),
                'rating': float(item.get('average_rating', 0.0)),
                'review_count': int(item.get('ratings_count', 0)),
                'isbn': item.get('isbn', ''),
                'isbn13': item.get('isbn13', ''),
                'authors': item.get('authors', []),
                'publisher': item.get('publisher', ''),
                'publication_info': {
                    'year': item.get('publication_year', ''),
                    'month': item.get('publication_month', ''),
                    'day': item.get('publication_day', '')
                },
                'description': item.get('description', ''),
                'language': item.get('language_code', ''),
                'source': 'goodreads',
                'type': item.get('type', '')
            }
        return item

    def _standardize_user(self, user: Dict) -> Dict:
        """Standardize user data based on source."""
        source = user.get('source', '')
        if source == 'yelp':
            return {
                'user_id': user['user_id'],
                'name': user.get('name', ''),
                'review_count': user.get('review_count', 0),
                'average_stars': user.get('average_stars', 0.0),
                'friends': user.get('friends', '').split(', ') if user.get('friends') else [],
                'fans': user.get('fans', 0),
                'elite': user.get('elite', ''),
                'stats': {
                    'useful': user.get('useful', 0),
                    'funny': user.get('funny', 0),
                    'cool': user.get('cool', 0)
                },
                'join_date': user.get('yelping_since', ''),
                'source': 'yelp'
            }
        elif source in ['amazon', 'goodreads']:
            # Amazon and Goodreads have minimal user info in the provided schema
            return {
                'user_id': user['user_id'],
                'source': source
            }
        return user

    def _standardize_review(self, review: Dict) -> Dict:
        """Standardize review data based on source."""
        source = review.get('source', '')
        if source == 'yelp':
            return {
                'review_id': review['review_id'],
                'user_id': review['user_id'],
                'item_id': review['item_id'],
                'rating': review.get('stars', 0.0),
                'text': review.get('text', ''),
                'date': review.get('date', ''),
                'votes': {
                    'useful': review.get('useful', 0),
                    'funny': review.get('funny', 0),
                    'cool': review.get('cool', 0)
                },
                'source': 'yelp',
                'type': review.get('type', '')
            }
        elif source == 'amazon':
            return {
                'review_id': review['review_id'],
                'user_id': review['user_id'],
                'item_id': review['item_id'],
                'rating': review.get('stars', 0.0),
                'title': review.get('title', ''),
                'text': review.get('text', ''),
                'date': review.get('timestamp', 0),
                'verified_purchase': review.get('verified_purchase', False),
                'helpful_votes': review.get('helpful_vote', 0),
                'images': review.get('images', []),
                'source': 'amazon',
                'type': review.get('type', '')
            }
        elif source == 'goodreads':
            return {
                'review_id': review['review_id'],
                'user_id': review['user_id'],
                'item_id': review['item_id'],
                'rating': review.get('stars', 0.0),
                'text': review.get('text', ''),
                'date_added': review.get('date_added', ''),
                'date_updated': review.get('date_updated', ''),
                'read_at': review.get('read_at', ''),
                'votes': review.get('n_votes', 0),
                'comments': review.get('n_comments', 0),
                'source': 'goodreads',
                'type': review.get('type', '')
            }
        return review

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch standardized user data based on user_id."""
        return self.user_data.get(user_id)

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch standardized item data based on item_id."""
        return self.item_data.get(item_id)

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch standardized reviews filtered by various parameters."""
        if review_id:
            return [self.review_data[review_id]] if review_id in self.review_data else []

        if item_id:
            return self.item_reviews.get(item_id, [])
        elif user_id:
            return self.user_reviews.get(user_id, [])

        return []

    def get_sources(self) -> Set[str]:
        """Get all unique data sources in the dataset."""
        sources = set()
        for item in self.item_data.values():
            sources.add(item.get('source', ''))
        return sources

    def get_items_by_source(self, source: str) -> List[Dict]:
        """Get all items from a specific source."""
        return [item for item in self.item_data.values() if item.get('source') == source]