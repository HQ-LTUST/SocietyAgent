import logging
import os
import json
import lmdb
from typing import Optional, Dict, List, Iterator, Any
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")


class CacheInteractionTool:
    """Enhanced Cache Interaction Tool with support for multiple data sources"""

    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing dataset files.
        """
        logger.info(f"Initializing CacheInteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir
        self.supported_sources = ['yelp', 'amazon', 'goodreads']

        # Create LMDB environments for each source
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        # Initialize environments for each source
        self.environments = {}
        for source in self.supported_sources:
            source_dir = os.path.join(self.env_dir, source)
            os.makedirs(source_dir, exist_ok=True)
            self.environments[source] = {
                'user': lmdb.open(os.path.join(source_dir, "users"), map_size=2 * 1024 * 1024 * 1024),
                'item': lmdb.open(os.path.join(source_dir, "items"), map_size=2 * 1024 * 1024 * 1024),
                'review': lmdb.open(os.path.join(source_dir, "reviews"), map_size=8 * 1024 * 1024 * 1024)
            }

        # Initialize all databases
        self._initialize_db()

    def get_user(self, user_id: str, source: Optional[str] = None) -> Optional[Dict]:
        """
        获取用户信息
        Args:
            user_id: str 用户ID
            source: Optional[str] 数据源标识 ('yelp', 'amazon', 'goodreads')
        Returns:
            Optional[Dict]: 用户信息字典，如果不存在则返回None
        """
        if source and source not in self.supported_sources:
            logger.warning(f"Unsupported source: {source}")
            return None

        if source:
            with self.environments[source]['user'].begin() as txn:
                user_data = txn.get(user_id.encode())
                return json.loads(user_data.decode()) if user_data else None
        else:
            # 如果没有指定source，尝试所有数据源
            for src in self.supported_sources:
                with self.environments[src]['user'].begin() as txn:
                    user_data = txn.get(user_id.encode())
                    if user_data:
                        return json.loads(user_data.decode())
        return None

    def get_item(self, item_id: str, source: Optional[str] = None) -> Optional[Dict]:
        """
        获取商品信息
        Args:
            item_id: str 商品ID
            source: Optional[str] 数据源标识 ('yelp', 'amazon', 'goodreads')
        Returns:
            Optional[Dict]: 商品信息字典，如果不存在则返回None
        """
        if source and source not in self.supported_sources:
            logger.warning(f"Unsupported source: {source}")
            return None

        if source:
            with self.environments[source]['item'].begin() as txn:
                item_data = txn.get(item_id.encode())
                return json.loads(item_data.decode()) if item_data else None
        else:
            # 如果没有指定source，尝试所有数据源
            for src in self.supported_sources:
                with self.environments[src]['item'].begin() as txn:
                    item_data = txn.get(item_id.encode())
                    if item_data:
                        return json.loads(item_data.decode())
        return None

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None,
            source: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch reviews filtered by various parameters.
        """
        if source and source not in self.supported_sources:
            logger.warning(f"Unsupported source: {source}")
            return []

        sources_to_check = [source] if source else self.supported_sources
        all_reviews = []

        for src in sources_to_check:
            with self.environments[src]['review'].begin() as txn:
                if review_id:
                    review_data = txn.get(review_id.encode())
                    if review_data:
                        return [json.loads(review_data)]
                elif item_id:
                    review_ids = json.loads(txn.get(f"item_{item_id}".encode()) or '[]')
                    for rid in review_ids:
                        review_data = txn.get(rid.encode())
                        if review_data:
                            all_reviews.append(json.loads(review_data))
                elif user_id:
                    review_ids = json.loads(txn.get(f"user_{user_id}".encode()) or '[]')
                    for rid in review_ids:
                        review_data = txn.get(rid.encode())
                        if review_data:
                            all_reviews.append(json.loads(review_data))

        return all_reviews

    def _get_source_from_file(self, filepath: str) -> str:
        """
        Determine the data source from file content or name.
        Args:
            filepath: Path to the data file
        Returns:
            str: Source identifier ('yelp', 'amazon', or 'goodreads')
        """
        try:
            # First try to determine from filename
            filename = os.path.basename(filepath).lower()
            for source in self.supported_sources:
                if source in filename:
                    return source

            # If can't determine from filename, check file content
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if 'source' in data:
                    source = data['source'].lower()
                    if source in self.supported_sources:
                        return source

            # Try to infer from content structure
            if 'asin' in data:
                return 'amazon'
            elif 'isbn' in data:
                return 'goodreads'
            elif 'yelping_since' in data:
                return 'yelp'

        except Exception as e:
            logger.warning(f"Error determining source from file {filepath}: {e}")

        # Default to yelp if can't determine
        return 'yelp'

    def _initialize_db(self):
        """Initialize all LMDB databases with data."""
        logger.info("开始初始化数据库...")

        # 明确定义文件类型映射
        file_type_mapping = {
            'user': 'user.json',
            'item': 'item.json',
            'review': 'review.json'
        }

        for db_type, filename in file_type_mapping.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"文件不存在: {filepath}")
                continue

            logger.info(f"处理文件: {filepath}")
            for source in self.supported_sources:
                if db_type == 'user':
                    self._initialize_user_db(filepath, source)
                elif db_type == 'item':
                    self._initialize_item_db(filepath, source)
                elif db_type == 'review':
                    self._initialize_review_db(filepath, source)

            self._verify_initialization()

    def _initialize_user_db(self, filepath: str, source: str):
        """Initialize user database for a specific source"""
        with self.environments[source]['user'].begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    count = 0
                    for user in self._iter_file(filepath):
                        if user.get('source') == source:
                            try:
                                cursor.put(
                                    user['user_id'].encode(),
                                    json.dumps(user).encode()
                                )
                                count += 1
                            except Exception as e:
                                logger.error(f"Error adding user {user.get('user_id', 'unknown')}: {e}")
                    logger.info(f"已加载 {count} 个 {source} 用户")

    def _initialize_item_db(self, filepath: str, source: str):
        """Initialize item database for a specific source"""
        with self.environments[source]['item'].begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    count = 0
                    for item in self._iter_file(filepath):
                        if item.get('source') == source:
                            try:
                                cursor.put(
                                    item['item_id'].encode(),
                                    json.dumps(item).encode()
                                )
                                count += 1
                            except Exception as e:
                                logger.error(f"Error adding item {item.get('item_id', 'unknown')}: {e}")
                    logger.info(f"已加载 {count} 个 {source} 商品")

    def _initialize_review_db(self, filepath: str, source: str):
        """Initialize review database for a specific source"""
        with self.environments[source]['review'].begin(write=True) as txn:
            if not txn.stat()['entries']:
                count = 0
                # 使用集合来防止重复
                processed_reviews = set()
                for review in self._iter_file(filepath):
                    if review.get('source') == source and review['review_id'] not in processed_reviews:
                        try:
                            processed_reviews.add(review['review_id'])
                            # Store the review
                            txn.put(
                                review['review_id'].encode(),
                                json.dumps(review).encode()
                            )

                            # Update item reviews index
                            item_key = f"item_{review['item_id']}".encode()
                            item_review_ids = set(json.loads(txn.get(item_key) or '[]'))
                            item_review_ids.add(review['review_id'])
                            txn.put(
                                item_key,
                                json.dumps(list(item_review_ids)).encode()
                            )

                            # Update user reviews index
                            user_key = f"user_{review['user_id']}".encode()
                            user_review_ids = set(json.loads(txn.get(user_key) or '[]'))
                            user_review_ids.add(review['review_id'])
                            txn.put(
                                user_key,
                                json.dumps(list(user_review_ids)).encode()
                            )
                            count += 1
                        except Exception as e:
                            logger.error(f"Error adding review {review.get('review_id', 'unknown')}: {e}")
                logger.info(f"已加载 {count} 条 {source} 评论")

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误 在文件 {filename} 第 {line_num} 行: {e}")
                    except Exception as e:
                        logger.error(f"处理错误 在文件 {filename} 第 {line_num} 行: {e}")
        except Exception as e:
            logger.error(f"打开文件 {filename} 时出错: {e}")

    def get_all_user_ids(self, source: Optional[str] = None) -> List[str]:
        """
        获取所有用户ID
        Args:
            source: Optional[str] 数据源标识 ('yelp', 'amazon', 'goodreads')
        Returns:
            List[str]: 用户ID列表
        """
        user_ids = []
        if source and source not in self.supported_sources:
            logger.warning(f"Unsupported source: {source}")
            return []

        sources_to_check = [source] if source else self.supported_sources

        for src in sources_to_check:
            with self.environments[src]['user'].begin() as txn:
                with txn.cursor() as cursor:
                    cursor.first()
                    while cursor.key():
                        user_ids.append(cursor.key().decode())
                        cursor.next()

        return user_ids

    def get_all_item_ids(self, source: Optional[str] = None) -> List[str]:
        """
        获取所有商品ID
        Args:
            source: Optional[str] 数据源标识 ('yelp', 'amazon', 'goodreads')
        Returns:
            List[str]: 商品ID列表
        """
        item_ids = []
        if source and source not in self.supported_sources:
            logger.warning(f"Unsupported source: {source}")
            return []

        sources_to_check = [source] if source else self.supported_sources

        for src in sources_to_check:
            with self.environments[src]['item'].begin() as txn:
                with txn.cursor() as cursor:
                    cursor.first()
                    while cursor.key():
                        item_ids.append(cursor.key().decode())
                        cursor.next()

        return item_ids

    def _verify_initialization(self):
        """验证数据库初始化状态"""
        logger.info("\n=== 验证数据库初始化状态 ===")
        for source in self.supported_sources:
            logger.info(f"\n检查 {source} 数据源:")
            for db_type in ['user', 'item', 'review']:
                with self.environments[source][db_type].begin() as txn:
                    stats = txn.stat()
                    logger.info(f"{db_type} 数据库条目数: {stats['entries']}")
                    # 验证数据是否可以正确读取
                    with txn.cursor() as cursor:
                        if cursor.first():
                            key, value = cursor.item()
                            try:
                                data = json.loads(value.decode())
                                logger.info(f"数据示例: {source} {db_type} - {key.decode()}")
                            except Exception as e:
                                logger.error(f"数据验证失败: {source} {db_type} - {e}")