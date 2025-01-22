from openai import OpenAI
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU, MemoryGenerative, MemoryTP, MemoryVoyager
from websocietysimulator.tools.interaction_tool import InteractionTool
from tools.evaluation_tool import UserModelingEvaluator # 更新导入路径更改为新的导入路径
from typing import Dict, List, Any, Optional, Union
from tools.cache_interaction_tool import CacheInteractionTool  # 假设文件名是 cache_interaction_tool.py
import logging
import os
import json


logger = logging.getLogger(__name__)


class OpenAIWrapper(LLMBase):
    """OpenAI wrapper that inherits from LLMBase"""

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI wrapper"""
        super().__init__(model=model)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.api_key = api_key
        self.base_url = base_url
        self._embedding_model = None

    def __call__(self, messages: List[Dict[str, str]],
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 stop_strs: Optional[List[str]] = None,
                 n: int = 1) -> Union[str, List[str]]:
        """Call OpenAI API to get response"""
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n
            )

            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise

    def get_embedding_model(self):
        """Get the embedding model"""
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._embedding_model


class OpenAIEmbeddings:
    """OpenAI Embeddings to replace InfinigenceEmbeddings"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def get_embeddings(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in OpenAI Embeddings API call: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """添加兼容原始接口的方法"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """添加用于查询的嵌入方法"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in query embedding: {e}")
            raise


from sklearn.ensemble import IsolationForest
import numpy as np


class MemoryManager:
    """Memory module manager with ML preprocessing"""

    def __init__(self, llm):
        """
        初始化 MemoryManager
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        # 初始化所有记忆模块
        self.memories = {
            'dilu': MemoryDILU(llm=llm),
            'generative': MemoryGenerative(llm=llm),
            'tp': MemoryTP(llm=llm),
            'voyager': MemoryVoyager(llm=llm)
        }
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.review_stats = []
        self.is_trained = False
        self.min_training_samples = 5  # 最小训练样本数

    def preprocess_review(self, review_data):
        """
        平台特定的评论数据预处理，提取特征
        Args:
            review_data: 评论数据字典
        Returns:
            numpy.array: 处理后的特征数组
        """
        try:
            source = review_data.get('source', 'yelp')

            # 基础特征
            base_features = [
                float(review_data.get('stars', 3.0)),  # 评分
                len(str(review_data.get('text', '')))  # 评论长度
            ]

            # 平台特定特征
            if source == 'yelp':
                text = review_data.get('text', '').lower()
                # Yelp特定关键词
                food_words = sum(word in text for word in [
                    'food', 'dish', 'taste', 'delicious', 'menu', 'portion',
                    'appetizer', 'entree', 'dessert', 'drink', 'flavor'
                ])
                service_words = sum(word in text for word in [
                    'service', 'staff', 'waiter', 'waitress', 'server', 'manager',
                    'attentive', 'friendly', 'rude', 'slow', 'quick'
                ])
                ambiance_words = sum(word in text for word in [
                    'atmosphere', 'ambiance', 'decor', 'noise', 'seating',
                    'comfortable', 'clean', 'dirty', 'busy', 'quiet'
                ])

                features = base_features + [
                    int(review_data.get('useful', 0)),
                    int(review_data.get('funny', 0)),
                    int(review_data.get('cool', 0)),
                    food_words,
                    service_words,
                    ambiance_words
                ]

            elif source == 'amazon':
                text = review_data.get('text', '').lower()
                # Amazon特定关键词
                product_words = sum(word in text for word in [
                    'quality', 'product', 'works', 'feature', 'durability',
                    'design', 'build', 'material', 'performance', 'reliability'
                ])
                technical_words = sum(word in text for word in [
                    'setup', 'installation', 'compatibility', 'technical',
                    'software', 'hardware', 'battery', 'charging', 'update'
                ])
                value_words = sum(word in text for word in [
                    'price', 'value', 'worth', 'cost', 'expensive', 'cheap',
                    'affordable', 'premium', 'budget', 'investment'
                ])

                features = base_features + [
                    int(review_data.get('helpful_vote', 0)),
                    product_words,
                    technical_words,
                    value_words,
                    0,  # placeholder
                    0   # placeholder
                ]

            elif source == 'goodreads':
                text = review_data.get('text', '').lower()
                # Goodreads特定关键词
                plot_words = sum(word in text for word in [
                    'plot', 'story', 'narrative', 'chapter', 'pacing',
                    'ending', 'twist', 'climax', 'resolution', 'arc'
                ])
                character_words = sum(word in text for word in [
                    'character', 'protagonist', 'development', 'motivation',
                    'personality', 'relationship', 'dialogue', 'interaction'
                ])
                writing_words = sum(word in text for word in [
                    'writing', 'prose', 'style', 'author', 'description',
                    'language', 'imagery', 'metaphor', 'theme', 'tone'
                ])

                features = base_features + [
                    int(review_data.get('n_votes', 0)),
                    plot_words,
                    character_words,
                    writing_words,
                    0,  # placeholder
                    0   # placeholder
                ]

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error in preprocess_review: {e}")
            return np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)

    def train_anomaly_detector(self, review_data_list):
        """
        训练异常检测器
        Args:
            review_data_list: 评论数据列表
        Returns:
            bool: 训练是否成功
        """
        try:
            if len(review_data_list) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(review_data_list)} samples")
                return False

            features_list = []
            for review in review_data_list:
                features = self.preprocess_review(review)
                features_list.append(features[0])

            if len(features_list) >= self.min_training_samples:
                self.isolation_forest.fit(features_list)
                self.is_trained = True
                logger.info(f"Anomaly detector trained with {len(features_list)} samples")
                return True
            return False

        except Exception as e:
            logger.error(f"Error in train_anomaly_detector: {e}")
            return False

    def select_memory_mode(self, review_data, business_data):
        """
        平台特定的记忆模式选择
        Args:
            review_data: 评论数据
            business_data: 商家/商品数据
        Returns:
            MemoryBase: 选择的记忆模块实例
        """
        try:
            source = review_data.get('source', 'yelp')

            # 如果review_data是字符串，尝试解析
            if isinstance(review_data, str):
                try:
                    review_data = json.loads(review_data)
                except:
                    logger.warning("Unable to parse review_data string, using default mode")
                    return self.memories['dilu']

            # 基本异常检测
            if not self.is_trained:
                logger.info("Using default DILU memory mode (model not trained)")
                return self.memories['dilu']

            features = self.preprocess_review(review_data)

            try:
                is_anomaly = self.isolation_forest.predict(features)[0] == -1
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                is_anomaly = False

            text = str(review_data.get('text', ''))

            # 平台特定逻辑
            if source == 'yelp':
                if is_anomaly:
                    logger.info("Anomaly detected in Yelp review, using Generative mode")
                    return self.memories['generative']

                # 餐厅评论通常需要更细节的记忆
                if (len(text) > 300 or
                    any(word in text.lower() for word in ['food', 'service', 'atmosphere', 'price'])):
                    logger.info("Detailed restaurant review detected, using Voyager mode")
                    return self.memories['voyager']

                if (isinstance(business_data, dict) and
                    (business_data.get('review_count', 0) > 500 or
                     business_data.get('is_chain', False))):
                    logger.info("Popular/chain restaurant detected, using TP mode")
                    return self.memories['tp']

            elif source == 'amazon':
                if is_anomaly:
                    logger.info("Anomaly detected in Amazon review, using Generative mode")
                    return self.memories['generative']

                # 技术产品评论需要更多技术细节记忆
                if (len(text) > 400 or
                    any(word in text.lower() for word in [
                        'technical', 'setup', 'quality', 'feature', 'performance'
                    ])):
                    logger.info("Technical product review detected, using Voyager mode")
                    return self.memories['voyager']

                if (isinstance(business_data, dict) and
                    (business_data.get('helpful_vote', 0) > 100 or
                     business_data.get('sales_rank', 0) < 1000)):
                    logger.info("Popular/high-ranking product detected, using TP mode")
                    return self.memories['tp']

            elif source == 'goodreads':
                if is_anomaly:
                    logger.info("Anomaly detected in Goodreads review, using Generative mode")
                    return self.memories['generative']

                # 书评通常需要更深入的分析记忆
                if (len(text) > 500 or
                    any(word in text.lower() for word in [
                        'plot', 'character', 'theme', 'writing', 'development'
                    ])):
                    logger.info("In-depth book review detected, using Voyager mode")
                    return self.memories['voyager']

                if (isinstance(business_data, dict) and
                    (business_data.get('n_votes', 0) > 50 or
                     business_data.get('is_bestseller', False))):
                    logger.info("Popular/bestseller book detected, using TP mode")
                    return self.memories['tp']

            logger.info("Using default DILU memory mode")
            return self.memories['dilu']

        except Exception as e:
            logger.error(f"Error in select_memory_mode: {e}")
            return self.memories['dilu']

    def add_memory(self, memory_type: str, current_situation: str):
        """
        添加记忆到指定类型的记忆模块
        Args:
            memory_type: 记忆模块类型
            current_situation: 当前场景
        """
        try:
            if memory_type in self.memories:
                self.memories[memory_type].addMemory(current_situation)
            else:
                logger.error(f"Unknown memory type: {memory_type}")
        except Exception as e:
            logger.error(f"Error in add_memory: {e}")

    def retrieve_memory(self, memory_type: str, query_scenario: str) -> str:
        """
        从指定类型的记忆模块检索记忆
        Args:
            memory_type: 记忆模块类型
            query_scenario: 查询场景
        Returns:
            str: 检索到的记忆
        """
        try:
            if memory_type in self.memories:
                memory_module = self.memories[memory_type]
                return memory_module.retriveMemory(query_scenario)
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return ""
        except Exception as e:
            logger.error(f"Error in retrieve_memory: {e}")
            return ""

    def get_all_memories(self, query_scenario: str) -> str:
        """
        从所有记忆模块检索记忆并合并结果
        Args:
            query_scenario: 查询场景
        Returns:
            str: 合并后的记忆内容
        """
        try:
            all_memories = []
            for memory_type, memory_module in self.memories.items():
                try:
                    memory = memory_module.retriveMemory(query_scenario)
                    if memory:
                        # 为不同来源的记忆添加标识
                        all_memories.append(f"[{memory_type}] {memory}")
                except Exception as e:
                    logger.error(f"Error retrieving memory from {memory_type}: {e}")
                    continue

            return "\n\n".join(all_memories) if all_memories else ""

        except Exception as e:
            logger.error(f"Error in get_all_memories: {e}")
            return ""

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, messages, temperature=0.1, **kwargs):
        """Override the parent class's __call__ method"""
        # Extract task description from messages
        task_description = messages[-1]["content"]

        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )

        return reasoning_result


class MySimulationAgent(SimulationAgent):
    """Optimized implementation of SimulationAgent with split API calls."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory_manager = MemoryManager(llm=self.llm)
        self.embedding = None
        self.interaction_tool = None
        self.task = None

        # Yelp特定的关键词集合
        self.yelp_keywords = {
            'food': [
                'food', 'dish', 'taste', 'delicious', 'menu', 'portion',
                'appetizer', 'entree', 'dessert', 'drink', 'flavor',
                'spicy', 'fresh', 'ingredients', 'cuisine', 'chef'
            ],
            'service': [
                'service', 'staff', 'waiter', 'waitress', 'server', 'manager',
                'attentive', 'friendly', 'rude', 'slow', 'quick', 'prompt',
                'helpful', 'professional', 'courteous', 'responsive'
            ],
            'ambiance': [
                'atmosphere', 'ambiance', 'decor', 'noise', 'seating',
                'comfortable', 'clean', 'dirty', 'busy', 'quiet', 'romantic',
                'cozy', 'modern', 'traditional', 'lighting', 'music'
            ]
        }

    def setup(self, interaction_tool, task, embedding_model):
        """Setup method to initialize other components after creation"""
        self.interaction_tool = interaction_tool
        self.task = task
        self.embedding = embedding_model

    def analyze_user(self, user_info: Dict) -> str:
        try:
            source = user_info.get('source', 'yelp')
            user_info_brief = str(user_info)[:1500]

            source_analysis_prompts = {
                'yelp': f'''
                       Analyze this Yelp user profile with focus on:
                       1. Restaurant Preferences:
                          - Favorite cuisines and dishes
                          - Price range preferences
                          - Location/neighborhood choices
                       2. Rating Pattern Analysis:
                          - Average rating: {user_info.get('average_stars', 0.0)}
                          - Rating distribution
                          - Rating justification style
                       3. Review Style:
                          - Detail level in descriptions
                          - Focus areas (food/service/ambiance)
                          - Writing tone and length
                       4. Elite Status Impact:
                          - Elite years: {user_info.get('elite', [])}
                          - Review quality progression
                       5. Review Frequency:
                          - Total reviews: {user_info.get('review_count', 0)}
                          - Visiting patterns
                          - Response to experiences
                       6. Special Interests:
                          - Dietary preferences
                          - Ambiance preferences
                          - Service expectations
                       ''',

                'amazon': f'''
                       Analyze this Amazon reviewer profile with focus on:
                       1. Product Category Preferences:
                          - Most reviewed categories
                          - Price sensitivity patterns
                          - Brand loyalty indicators
                       2. Review Credibility:
                          - Verified purchases ratio
                          - Helpful votes received: {user_info.get('helpful_votes_received', 0)}
                          - Review quality consistency
                       3. Technical Detail Level:
                          - Specification focus
                          - Use case descriptions
                          - Comparison frequency
                       4. Purchase Behavior:
                          - Review timing vs purchase
                          - Long-term usage updates
                          - Return frequency mentions
                       5. Review Impact:
                          - Average helpfulness ratio: {user_info.get('helpful_ratio', 0.0)}
                          - Review length patterns
                       ''',

                'goodreads': f'''
                       Analyze this Goodreads user profile with focus on:
                       1. Reading Preferences:
                          - Favorite genres
                          - Author loyalty patterns
                          - Series completion rate
                       2. Review Depth:
                          - Plot analysis style
                          - Character discussion depth
                          - Writing style focus
                       3. Rating Patterns:
                          - Average rating: {user_info.get('average_rating', 0.0)}
                          - Genre-specific ratings
                          - Rating justification style
                       4. Review Impact:
                          - Total books reviewed: {user_info.get('books_reviewed', 0)}
                          - Community engagement
                          - Influence patterns
                       5. Reading Habits:
                          - Reading frequency
                          - Genre diversity
                          - Review timing patterns
                       '''
            }

            task = f'''
               {source_analysis_prompts.get(source, "")}
               Profile data: {user_info_brief}
               Generate a detailed analysis focusing on review and rating patterns.
               '''

            messages = [{"role": "user", "content": task}]
            return self.reasoning(messages)

        except Exception as e:
            logger.error(f"Error in analyze_user: {e}")
            return f"Average {source} reviewer"

    def analyze_business(self, business_info: Dict) -> str:
        """
        Analyze business information based on source type.

        Args:
            business_info (Dict): Dictionary containing business information with source type

        Returns:
            str: Analysis result from the reasoning module
        """
        try:
            if not business_info:
                logger.error("Empty business_info provided")
                return ""

            source = business_info.get('source')
            if not source:
                logger.error("Source not specified in business_info")
                return ""

            # Define source-specific prompts and required fields
            source_prompts = {
                'yelp': {
                    'title': 'restaurant',
                    'sections': {
                        'Core Restaurant Profile': {
                            'Name': business_info.get('name', 'Unknown'),
                            'Cuisine Type': business_info.get('categories', []),
                            'Price Range': business_info.get('price_range', 'Unknown'),
                            'Location': business_info.get('neighborhood', 'Unknown')
                        },
                        'Menu and Food': [
                            'Signature Dishes',
                            'Menu Variety',
                            'Special Dietary Options',
                            'Portion Sizes',
                            'Price-Value Ratio'
                        ],
                        'Service Standards': [
                            'Service Style',
                            'Staff Professionalism',
                            'Peak Hours Management',
                            'Special Requests Handling'
                        ],
                        'Ambiance and Setup': [
                            'Restaurant Design',
                            'Seating Arrangement',
                            'Noise Level',
                            'Cleanliness Standards'
                        ],
                        'Additional Features': [
                            'Parking Availability',
                            'Accessibility',
                            'Outdoor Seating',
                            'Special Amenities'
                        ],
                        'Performance Metrics': {
                            'Overall Rating': business_info.get('stars', 0.0),
                            'Review Count': business_info.get('review_count', 0),
                            'Popular Times': None,
                            'Customer Return Rate': None
                        }
                    }
                },
                'amazon': {
                    'title': 'product',
                    'sections': {
                        'Product Details': {
                            'Title': business_info.get('title', 'Unknown'),
                            'Category': business_info.get('main_category', 'Unknown'),
                            'Brand': business_info.get('brand', 'Unknown'),
                            'Price': business_info.get('price', 'Unknown')
                        },
                        'Technical Specifications': [
                            'Key features',
                            'Technical parameters',
                            'Compatibility info',
                            'Product dimensions',
                            'Materials used'
                        ],
                        'Usage Patterns': [
                            'Common use cases',
                            'User experiences',
                            'Durability reports',
                            'Maintenance requirements',
                            'User skill level needed'
                        ],
                        'Value Proposition': [
                            'Price positioning',
                            'Competitive advantages',
                            'Cost-benefit analysis',
                            'Warranty coverage',
                            'After-sales support'
                        ],
                        'Quality Indicators': [
                            'Manufacturing quality',
                            'Reliability patterns',
                            'Common issues',
                            'Quality control measures',
                            'Product lifespan'
                        ],
                        'Market Performance': {
                            'Average Rating': business_info.get('average_rating', 0.0),
                            'Review Count': business_info.get('review_count', 0),
                            'Sales Rank': business_info.get('sales_rank', 'Unknown'),
                            'Customer Satisfaction': None,
                            'Market Position': None
                        }
                    }
                },
                'goodreads': {
                    'title': 'book',
                    'sections': {
                        'Book Details': {
                            'Title': business_info.get('title', 'Unknown'),
                            'Author': business_info.get('author', 'Unknown'),
                            'Genre': business_info.get('genre', 'Unknown'),
                            'Publication Year': business_info.get('publication_year', 'Unknown'),
                            'Publisher': business_info.get('publisher', 'Unknown')
                        },
                        'Content Analysis': [
                            'Plot structure',
                            'Character development',
                            'Writing style',
                            'Narrative pace',
                            'Story complexity'
                        ],
                        'Reader Experience': [
                            'Engagement level',
                            'Emotional impact',
                            'Reading difficulty',
                            'Target audience',
                            'Reading time estimate'
                        ],
                        'Series Context': [
                            'Series position',
                            'Story continuity',
                            'Previous books impact',
                            'Series completion status',
                            'Series popularity'
                        ],
                        'Literary Elements': [
                            'Themes exploration',
                            'Narrative techniques',
                            'World-building',
                            'Character arcs',
                            'Symbolism usage'
                        ],
                        'Reception': {
                            'Average Rating': business_info.get('average_rating', 0.0),
                            'Review Count': business_info.get('review_count', 0),
                            'Awards': business_info.get('awards', []),
                            'Critical Reception': None,
                            'Reader Demographics': None
                        }
                    }
                }
            }

            if source not in source_prompts:
                logger.error(f"Unsupported source: {source}")
                return ""

            # Generate unified prompt format from structure
            prompt = f"Analyze this {source_prompts[source]['title']} focusing on:\n\n"

            for section_num, (section, content) in enumerate(source_prompts[source]['sections'].items(), 1):
                prompt += f"{section_num}. {section}:\n"

                if isinstance(content, dict):
                    # Handle dictionary type content (key-value pairs)
                    for key, value in content.items():
                        if value is not None:  # Only include non-None values
                            formatted_value = value if value != [] else 'Not specified'
                            prompt += f"   - {key}: {formatted_value}\n"

                elif isinstance(content, list):
                    # Handle list type content (bullet points)
                    for item in content:
                        prompt += f"   - {item}\n"

                prompt += "\n"  # Add extra newline between sections

            # Add source-specific analysis instructions
            source_instructions = {
                'yelp': '''
    Please analyze this restaurant's profile comprehensively, focusing on:
    - The restaurant's market positioning based on cuisine type and price range
    - The likely customer experience across food, service, and ambiance
    - Key strengths and potential areas for improvement
    - Competitive advantages in its location and category''',

                'amazon': '''
    Please analyze this product's profile comprehensively, focusing on:
    - The product's market positioning and target customer segment
    - Key competitive advantages and unique selling points
    - Primary use cases and customer benefits
    - Potential concerns or limitations based on reviews and specifications''',

                'goodreads': '''
    Please analyze this book's profile comprehensively, focusing on:
    - The book's literary significance and target audience
    - Key themes and writing style characteristics
    - Reader engagement and emotional impact
    - Position within its genre and comparison to similar works'''
            }

            # Add the source-specific instructions to the prompt
            prompt += source_instructions.get(source, '')

            # Create messages for reasoning
            messages = [{"role": "user", "content": prompt.strip()}]

            # Call reasoning module and return results
            return self.reasoning(messages)

        except Exception as e:
            logger.error(f"Error in analyze_business: {e}")
            logger.debug(f"Business info: {business_info}")  # Add detailed debug info
            return ""

    def process_reviews(self, reviews: List[Dict], source: str) -> str:
        """Platform-specific review processing"""
        try:
            if not reviews:
                return "No previous reviews available."

            sample_reviews = reviews[:5]
            ratings = [r.get('stars', 0) for r in sample_reviews]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            lengths = [len(str(r.get('text', ''))) for r in sample_reviews]
            avg_length = sum(lengths) / len(lengths) if lengths else 0

            source_specific_prompts = {
                'yelp': f'''
                       Analyze patterns in these Yelp reviews focusing on:
                       1. Food Quality Patterns:
                          - Common dish mentions
                          - Taste descriptions
                          - Portion feedback
                          - Price-value comments
                       2. Service Experience:
                          - Staff interaction patterns
                          - Wait time feedback
                          - Problem resolution
                          - Special requests handling
                       3. Atmosphere Comments:
                          - Ambiance descriptions
                          - Noise level mentions
                          - Seating comfort
                          - Cleanliness notes
                       4. Overall Patterns:
                          - Average rating: {avg_rating:.1f}
                          - Average length: {avg_length:.0f} characters
                          - Common praises
                          - Frequent criticisms
                       ''',

                'amazon': f'''
                       Analyze Amazon review patterns:
                       1. Product Performance:
                          - Functionality feedback
                          - Reliability comments
                          - Quality observations
                       2. Technical Details:
                          - Specification accuracy
                          - Feature usability
                          - Compatibility notes
                       3. Usage Experience:
                          - Installation process
                          - Learning curve
                          - Daily usage
                       4. Value Analysis:
                          - Price justification
                          - Competitive comparison
                          - Long-term value
                       Average rating: {avg_rating:.1f}
                       Average length: {avg_length:.0f} characters
                       ''',

                'goodreads': f'''
                       Analyze Goodreads review patterns:
                       1. Story Elements:
                          - Plot development
                          - Character analysis
                          - Writing style comments
                       2. Reader Engagement:
                          - Emotional impact
                          - Page-turner mentions
                          - Reading experience
                       3. Literary Analysis:
                          - Theme discussion
                          - Symbolism notes
                          - Style appreciation
                       4. Comparative Notes:
                          - Genre comparisons
                          - Author style
                          - Series context
                       Average rating: {avg_rating:.1f}
                       Average length: {avg_length:.0f} characters
                       '''
            }

            source_metrics = {
                'yelp': lambda r: f"Useful: {r.get('useful', 0)}, Funny: {r.get('funny', 0)}, Cool: {r.get('cool', 0)}",
                'amazon': lambda r: f"Helpful votes: {r.get('helpful_vote', 0)}",
                'goodreads': lambda r: f"Votes: {r.get('n_votes', 0)}"
            }

            metrics_function = source_metrics.get(source, lambda r: "")
            samples = [(r.get('text', '')[:100] + '... | ' + metrics_function(r)) for r in sample_reviews[:3]]

            task = f'''
               {source_specific_prompts.get(source, "")}
               Sample reviews with metrics:
               {samples}

               Analyze these patterns for generating authentic platform-specific reviews.
               '''

            messages = [{"role": "user", "content": task}]
            return self.reasoning(messages)

        except Exception as e:
            logger.error(f"Error in process_reviews: {e}")
            return "Error analyzing review patterns."

    def generate_review(self, user_summary: str, business_summary: str, review_context: str, source: str) -> str:
        try:
            # 添加长度控制
            length_guidelines = {
                'yelp': (50, 150),  # 最短50字符，最长150字符
                'amazon': (30, 100),  # 最短30字符，最长100字符
                'goodreads': (20, 80)  # 最短20字符，最长80字符
            }
            min_len, max_len = length_guidelines.get(source, (30, 100))

            if source == 'yelp':
                task_description = f'''
                Task: Generate a concise Yelp restaurant review based on the following context.
                IMPORTANT: Keep the review between {min_len} and {max_len} characters. Be direct and to the point.

                USER PROFILE SUMMARY:
                {user_summary[:300]}

                RESTAURANT DETAILS:
                {business_summary[:300]}

                REVIEW PATTERNS:
                {review_context[:300]}

                Required Output Format:
                stars: [1.0-5.0]
                useful: [0-3]
                funny: [0-1]
                cool: [0-2]
                review: [Your concise review text]

                Style Guidelines:
                - Be brief and direct
                - Focus on 1-2 key aspects only
                - Use natural, casual language
                - Avoid unnecessary details
                - Write like a real person, not a professional critic
                '''
            elif source == 'amazon':
                task_description = f'''
                Task: Generate a short Amazon product review based on provided context.
                IMPORTANT: Keep the review between {min_len} and {max_len} characters. Focus on the main points only.

                USER PROFILE SUMMARY:
                {user_summary[:300]}

                PRODUCT DETAILS:
                {business_summary[:300]}

                REVIEW PATTERNS:
                {review_context[:300]}

                Required Output Format:
                stars: [1.0-5.0]
                useful: [0-5]
                review: [Your concise review text]

                Style Guidelines:
                - Get straight to the point
                - Focus on main product experience
                - Use casual language
                - Keep it simple and direct
                '''
            else:  # goodreads
                task_description = f'''
                Task: Generate a brief Goodreads book review based on provided context.
                IMPORTANT: Keep the review between {min_len} and {max_len} characters. Be concise and direct.

                USER PROFILE SUMMARY:
                {user_summary[:300]}

                BOOK DETAILS:
                {business_summary[:300]}

                REVIEW PATTERNS:
                {review_context[:300]}

                Required Output Format:
                stars: [1.0-5.0]
                useful: [0-5]
                review: [Your concise review text]

                Style Guidelines:
                - Share quick overall impression
                - Avoid detailed plot summaries
                - Use natural language
                - Write like a casual reader
                '''

            messages = [{"role": "user", "content": task_description}]
            result = self.reasoning(
                messages=messages,
                temperature=0.0,
                max_tokens=200  # 限制输出长度
            )

            if 'stars:' not in result or 'review:' not in result:
                logger.warning("Generated review missing required fields, using default format")
                default_result = f'''
                stars: 3.0
                useful: 1
                {'funny: 0\ncool: 0' if source == 'yelp' else ''}
                review: Short review not available.
                '''
                return default_result

            return result

        except Exception as e:
            logger.error(f"Error in generate_review: {e}")
            return f'''
            stars: 3.0
            useful: 1
            {'funny: 0\ncool: 0' if source == 'yelp' else ''}
            review: Error generating short review.
            '''

    def parse_review_result(self, result: str, source: str) -> tuple:
        """
        解析生成的评论结果
        Args:
            result: 生成的评论文本
            source: 数据源
        Returns:
            tuple: (stars, useful, funny, cool, review_text)
        """
        try:
            lines = result.strip().split('\n')
            parsed = {}
            current_field = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                for field in ['stars:', 'useful:', 'funny:', 'cool:', 'review:']:
                    if line.lower().startswith(field):
                        current_field = field[:-1]
                        value = line[len(field):].strip()
                        parsed[current_field] = value
                        break
                else:
                    if current_field == 'review':
                        parsed['review'] = parsed.get('review', '') + ' ' + line

            # 转换值
            stars = float(parsed.get('stars', '3.0'))
            stars = max(1.0, min(5.0, stars))

            useful = int(float(parsed.get('useful', '0')))
            funny = cool = 0

            if source == 'yelp':
                useful = max(0, min(3, useful))
                funny = int(float(parsed.get('funny', '0')))
                funny = max(0, min(1, funny))
                cool = int(float(parsed.get('cool', '0')))
                cool = max(0, min(2, cool))
            else:
                useful = max(0, min(5, useful))

            review_text = parsed.get('review', 'No review content provided.').strip()

            return stars, useful, funny, cool, review_text

        except Exception as e:
            logger.error(f"Error parsing review result: {e}")
            return 3.0, 0, 0, 0, "Error processing review."

    def workflow(self):
        """
        执行评论模拟工作流
        Returns:
            tuple: (stars, useful, funny, cool, review_text)
        """
        try:
            source = self.task.get('source', 'yelp')

            # 获取基础信息
            user = self.interaction_tool.get_user(
                user_id=self.task['user_id'],
                source=source
            )
            business = self.interaction_tool.get_item(
                item_id=self.task['item_id'],
                source=source
            )

            if not user or not business:
                logger.error(f"Failed to get user or business data for task {self.task}")
                return 3.0, 0, 0, 0, "Error: Unable to retrieve user or business data."

            logger.info("\n====== Input Data ======")
            logger.info(f"User ID: {self.task['user_id']}")
            logger.info(f"Business ID: {self.task['item_id']}")
            logger.info(f"Source: {source}")

            # 获取评论数据
            reviews_item = self.interaction_tool.get_reviews(
                item_id=self.task['item_id'],
                source=source
            )
            reviews_user = self.interaction_tool.get_reviews(
                user_id=self.task['user_id'],
                source=source
            )

            # 训练异常检测器
            all_reviews = []
            if reviews_item:
                all_reviews.extend(reviews_item[:10])
            if reviews_user:
                all_reviews.extend(reviews_user[:5])

            if all_reviews:
                self.memory_manager.train_anomaly_detector(all_reviews)

            # 添加到记忆模块
            if source == 'yelp':
                # 增强 Yelp 评论的记忆处理
                if reviews_item:
                    for review in reviews_item[:5]:
                        text = review.get('text', '').lower()

                        # 计算特征词出现频率
                        food_score = sum(word in text for word in self.yelp_keywords['food'])
                        service_score = sum(word in text for word in self.yelp_keywords['service'])
                        ambiance_score = sum(word in text for word in self.yelp_keywords['ambiance'])

                        # 根据关键词分析重要性
                        importance_score = food_score + service_score + ambiance_score

                        memory_mode = self.memory_manager.select_memory_mode(review, business)
                        if importance_score > 5:  # 高价值评论
                            memory_mode.addMemory(f"HIGH QUALITY restaurant review: {review.get('text', '')}")
                        else:  # 普通评论
                            memory_mode.addMemory(f"regular review: {review.get('text', '')}")

                if reviews_user:
                    for review in reviews_user[:3]:
                        text = review.get('text', '').lower()
                        food_score = sum(word in text for word in self.yelp_keywords['food'])
                        service_score = sum(word in text for word in self.yelp_keywords['service'])

                        memory_mode = self.memory_manager.select_memory_mode(review, business)
                        if food_score > 2 or service_score > 2:  # 用户关注点明显的评论
                            memory_mode.addMemory(f"USER PREFERENCE review: {review.get('text', '')}")
                        else:
                            memory_mode.addMemory(f"user review: {review.get('text', '')}")
            else:
                # 其他平台的标准处理
                if reviews_item:
                    for review in reviews_item[:5]:
                        memory_mode = self.memory_manager.select_memory_mode(review, business)
                        memory_mode.addMemory(f"item review: {review.get('text', '')}")

                if reviews_user:
                    for review in reviews_user[:3]:
                        memory_mode = self.memory_manager.select_memory_mode(review, business)
                        memory_mode.addMemory(f"user review: {review.get('text', '')}")

            # 分析组件
            user_summary = self.analyze_user(user)
            business_summary = self.analyze_business(business)
            review_patterns = self.process_reviews(reviews_item, source)

            # 获取所有相关记忆
            all_memories = self.memory_manager.get_all_memories(str(business))

            # 生成评论
            result = self.generate_review(
                user_summary=user_summary,
                business_summary=business_summary,
                review_context=f"{review_patterns}\n{all_memories}",
                source=source
            )

            try:
                # 使用统一的解析方法
                stars, useful, funny, cool, review_text = self.parse_review_result(result, source)

                # 记录生成的评论详情
                logger.info("\n====== Generated Review ======")
                logger.info(f"Stars: {stars}")
                logger.info(f"Useful: {useful}")
                if source == 'yelp':
                    logger.info(f"Funny: {funny}")
                    logger.info(f"Cool: {cool}")
                logger.info(f"Review: {review_text}")

                # 获取并打印真实评论用于对比
                real_reviews = self.interaction_tool.get_reviews(
                    user_id=self.task['user_id'],
                    item_id=self.task['item_id'],
                    source=source
                )

                if real_reviews:
                    logger.info("\n====== Real Review ======")
                    real_review = real_reviews[0]
                    logger.info(f"Stars: {real_review.get('stars', 0.0)}")
                    logger.info(f"Useful: {real_review.get('useful', 0)}")
                    if source == 'yelp':
                        logger.info(f"Funny: {real_review.get('funny', 0)}")
                        logger.info(f"Cool: {real_review.get('cool', 0)}")
                    logger.info(f"Review: {real_review.get('text', '')}")

                logger.info("\n====== End of Review Generation ======\n")
                return stars, useful, funny, cool, review_text

            except Exception as e:
                logger.error(f"Error in review generation: {e}")
                return 3.0, 0, 0, 0, "Error generating review."

        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return 3.0, 0, 0, 0, "Error in review generation process."


def main(batch_size: int = 50, max_tasks: int = None):
    """
    执行主要评估流程
    Args:
        batch_size: 每个批次的任务数量
        max_tasks: 最大任务数量限制
    Returns:
        tuple: (all_simulated_data, all_real_data)
    """
    all_simulated_data = []
    all_real_data = []

    try:
        # 创建 OpenAI wrapper
        llm = OpenAIWrapper(
            api_key="sk-cS3K2urP0tyj470rD0Fb0a61EaC949D8AcAfC7C55eD00dCc",
            base_url="https://api.gpt.ge/v1/",
            model="gpt-3.5-turbo"
        )

        # 设置数据目录和创建交互工具
        data_directory = r"C:\Users\hq200\Desktop\User Modeling Track\files"
        interaction_tool = CacheInteractionTool(data_dir=data_directory)

        # 创建评估器
        evaluator = UserModelingEvaluator(device="auto")

        # 创建任务列表
        tasks = []
        logger.info("正在收集所有数据源的任务...")

        # 按数据源分别处理
        for source in ['yelp', 'amazon', 'goodreads']:
            users = interaction_tool.get_all_user_ids(source=source)
            items = interaction_tool.get_all_item_ids(source=source)
            logger.info(f"{source} 数据源: {len(users)} 个用户, {len(items)} 个商品")

            task_count = 0
            for user_id in users:
                user_reviews = interaction_tool.get_reviews(user_id=user_id, source=source)
                if user_reviews:
                    for review in user_reviews:
                        item_id = review['item_id']
                        tasks.append({
                            "user_id": user_id,
                            "item_id": item_id,
                            "source": source
                        })
                        task_count += 1

                        if max_tasks is not None and task_count >= max_tasks / 3:
                            break

                if max_tasks is not None and task_count >= max_tasks / 3:
                    break

        total_tasks = len(tasks)
        logger.info(f"总任务数: {total_tasks}")

        # 批量处理任务
        for i, task in enumerate(tasks, 1):
            try:
                source = task['source']
                logger.info(f"\n处理 {source} 任务 {i}/{total_tasks}")
                logger.info(f"用户 ID: {task['user_id']}, 商品 ID: {task['item_id']}")

                # 创建和设置 agent
                agent = MySimulationAgent(llm=llm)
                agent.setup(
                    interaction_tool=interaction_tool,
                    task=task,
                    embedding_model=llm.get_embedding_model()
                )

                # 执行 workflow
                stars, useful, funny, cool, review_text = agent.workflow()

                # 打印模拟生成的评论
                logger.info("\n=== 模拟生成的评论 ===")
                logger.info(f"评分: {stars:.1f}")
                logger.info(f"评论内容: {review_text}\n")

                # 收集模拟数据
                simulated_data = {
                    'stars': stars,
                    'review': review_text,
                    'user_id': task['user_id'],
                    'item_id': task['item_id'],
                    'source': source,
                }

                # 根据源添加特定字段
                if source == 'yelp':
                    simulated_data.update({
                        'useful': useful,
                        'funny': funny,
                        'cool': cool
                    })
                elif source == 'amazon':
                    simulated_data.update({
                        'helpful_vote': useful
                    })
                elif source == 'goodreads':
                    simulated_data.update({
                        'n_votes': useful
                    })

                all_simulated_data.append(simulated_data)

                # 获取真实数据
                real_reviews = interaction_tool.get_reviews(
                    user_id=task['user_id'],
                    item_id=task['item_id'],
                    source=source
                )

                if real_reviews:
                    real_review = real_reviews[0]
                    # 打印真实评论
                    logger.info("=== 真实评论 ===")
                    logger.info(f"评分: {real_review.get('stars', 0.0):.1f}")
                    logger.info(f"评论内容: {real_review.get('text', '')}\n")

                    # 收集真实数据
                    review_data = {
                        'stars': real_review.get('stars', 0.0),
                        'review': real_review.get('text', ''),
                        'user_id': task['user_id'],
                        'item_id': task['item_id'],
                        'source': source
                    }

                    # 根据源添加特定字段
                    if source == 'yelp':
                        review_data.update({
                            'useful': real_review.get('useful', 0),
                            'funny': real_review.get('funny', 0),
                            'cool': real_review.get('cool', 0)
                        })
                    elif source == 'amazon':
                        review_data.update({
                            'helpful_vote': real_review.get('helpful_vote', 0)
                        })
                    elif source == 'goodreads':
                        review_data.update({
                            'n_votes': real_review.get('n_votes', 0)
                        })

                    all_real_data.append(review_data)

                # 定期评估
                if i % batch_size == 0:
                    for eval_source in ['yelp', 'amazon', 'goodreads']:
                        source_simulated = [d for d in all_simulated_data if d['source'] == eval_source]
                        source_real = [d for d in all_real_data if d['source'] == eval_source]

                        if source_simulated and source_real:
                            metrics = evaluator.evaluate(
                                simulated_data=source_simulated,
                                real_data=source_real
                            )

                            logger.info(f"\n{eval_source.upper()} 评估指标 (批次 {i // batch_size + 1}):")
                            logger.info(f"样本数量: {len(source_simulated)}")
                            logger.info(f"偏好估计: {metrics.preference_estimation:.4f}")
                            logger.info(f"评论生成: {metrics.review_generation:.4f}")
                            logger.info(f"整体质量: {metrics.overall_quality:.4f}")

            except Exception as e:
                logger.error(f"处理任务 {i} 时出错: {str(e)}")
                logger.error("错误详情:", exc_info=True)
                continue

    except Exception as e:
        logger.error(f"main 函数执行出错: {str(e)}")
        logger.error("错误详情:", exc_info=True)
        raise

    finally:
        # 最终评估
        logger.info("\n=== 最终评估结果 ===")
        try:
            for source in ['yelp', 'amazon', 'goodreads']:
                source_simulated = [d for d in all_simulated_data if d['source'] == source]
                source_real = [d for d in all_real_data if d['source'] == source]

                if source_simulated and source_real:
                    metrics = evaluator.evaluate(
                        simulated_data=source_simulated,
                        real_data=source_real
                    )

                    logger.info(f"\n{source.upper()} 最终评估指标:")
                    logger.info(f"样本数量: {len(source_simulated)}")
                    logger.info(f"偏好估计: {metrics.preference_estimation:.4f}")
                    logger.info(f"评论生成: {metrics.review_generation:.4f}")
                    logger.info(f"整体质量: {metrics.overall_quality:.4f}")

            # 保存结果到文件
            with open('simulated_reviews.json', 'w', encoding='utf-8') as f:
                json.dump(all_simulated_data, f, ensure_ascii=False, indent=2)

            with open('real_reviews.json', 'w', encoding='utf-8') as f:
                json.dump(all_real_data, f, ensure_ascii=False, indent=2)

            logger.info("\n结果已保存到 simulated_reviews.json 和 real_reviews.json")

        except Exception as e:
            logger.error(f"最终评估或保存结果时出错: {str(e)}")
            logger.error("错误详情:", exc_info=True)

        finally:
            return all_simulated_data, all_real_data


if __name__ == "__main__":
    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    all_simulated_data, all_real_data = main(batch_size=10,  max_tasks=5)


