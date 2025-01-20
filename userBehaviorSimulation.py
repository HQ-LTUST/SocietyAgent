from openai import OpenAI
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from websocietysimulator.tools.interaction_tool import InteractionTool
from tools.evaluation_tool import SimulationEvaluator  # 更新导入路径更改为新的导入路径
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

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
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
        self.memory = MemoryDILU(llm=self.llm)
        self.embedding = None

    def setup(self, interaction_tool, task, embedding_model):
        """Setup method to initialize other components after creation"""
        self.interaction_tool = interaction_tool
        self.task = task
        self.embedding = embedding_model

    def analyze_user(self, user_info: str) -> str:
        """Analyze user profile in a separate API call with text chunking"""
        try:
            # 限制输入长度，只取前1000字符的基本信息
            user_info_brief = str(user_info)[:1000]

            task = f'''
            Analyze this partial Yelp user profile summary. Focus on:
            - Rating style
            - Review preferences
            Keep it very brief.

            Profile excerpt: {user_info_brief}
            '''
            messages = [{"role": "user", "content": task}]
            user_analysis = self.reasoning(messages)
            return user_analysis
        except Exception as e:
            logger.error(f"Error in analyze_user: {e}")
            return "Average Yelp reviewer"  # 出错时返回默认值而不是空字符串

    def analyze_business(self, business_info: str) -> str:
        """Analyze business profile in a separate API call"""
        try:
            task = f'''
            You are analyzing a business profile from Yelp. Provide a brief summary of:
            - Key features and offerings
            - Notable characteristics
            - Main aspects customers might review
            Keep it concise and factual.

            Business: {business_info}
            '''
            messages = [{"role": "user", "content": task}]
            business_analysis = self.reasoning(messages)
            return business_analysis
        except Exception as e:
            logger.error(f"Error in analyze_business: {e}")
            return ""

    def process_reviews(self, reviews: list) -> str:
        """Process a sample of reviews in a separate API call"""
        if not reviews:
            return ""

        try:
            # Take first review and limit length
            sample_review = reviews[0]['text'][:300]  # Limit length to control tokens

            task = f'''
            Analyze this sample review and extract:
            - Key points mentioned
            - Tone and style
            Keep the analysis brief.

            Review: {sample_review}
            '''
            messages = [{"role": "user", "content": task}]
            review_analysis = self.reasoning(messages)
            return review_analysis
        except Exception as e:
            logger.error(f"Error in process_reviews: {e}")
            return ""

    def generate_review(self, user_summary: str, business_summary: str, review_context: str) -> str:
        """Generate the final review based on analyzed components"""
        try:
            task_description = f'''
            You are writing a Yelp review based on these insights:

            User Analysis: {user_summary[:500]}
            Business Summary: {business_summary[:500]}
            Review Context: {review_context[:500]}

            Write a review that:
            - Matches the user's typical style
            - Focuses on key business aspects
            - Has appropriate rating and engagement metrics

            Format:
            stars: [1.0-5.0]
            useful: [count]
            funny: [count] 
            cool: [count]
            review: [2-4 sentences]
            '''

            messages = [{"role": "user", "content": task_description}]
            return self.reasoning(messages)
        except Exception as e:
            logger.error(f"Error in generate_review: {e}")
            return ""

    def workflow(self):
        """
        Execute the review simulation with memory module integration
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            # Basic info gathering
            user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
            business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))

            # Get real review data for comparison
            real_reviews = self.interaction_tool.get_reviews(
                user_id=self.task['user_id'],
                item_id=self.task['item_id']
            )

            logger.info("\n====== Input Data ======")
            logger.info(f"User ID: {self.task['user_id']}")
            logger.info(f"Business ID: {self.task['item_id']}")

            # Get all relevant reviews
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])

            # Add reviews to memory
            logger.info("Adding reviews to memory...")
            if reviews_item:
                for review in reviews_item[:5]:  # Only use recent 5 reviews
                    self.memory(f"review: {review['text']}")

            if reviews_user:
                for review in reviews_user[:3]:  # Only use recent 3 user reviews
                    self.memory(f"review: {review['text']}")

            # User Analysis
            user_info_brief = str(user)[:1000]
            task = f'''
            Analyze this Yelp user's reviewing style briefly.
            Profile: {user_info_brief}
            Focus only on: typical ratings, writing style, and preferences.
            Keep response under 100 words.
            '''
            messages = [{"role": "user", "content": task}]
            user_summary = self.reasoning(messages)

            # Business Analysis
            business_info_brief = str(business)[:1000]
            task = f'''
            Analyze this business profile briefly.
            Business: {business_info_brief}
            Focus only on: key features and main review points.
            Keep response under 100 words.
            '''
            messages = [{"role": "user", "content": task}]
            business_summary = self.reasoning(messages)

            # Retrieve similar reviews from memory
            logger.info("Retrieving similar reviews from memory...")
            similar_reviews = self.memory(str(business))

            # Generate Final Review
            task_description = f'''
            Write a Yelp review following EXACTLY this format:

            stars: [rating between 1.0 and 5.0]
            useful: [number between 0 and 3]
            funny: [number between 0 and 1]
            cool: [number between 0 and 2]
            review: [Your detailed review here]

            Context:
            - User style: {user_summary[:200]}
            - Business info: {business_summary[:200]}
            - Similar reviews: {similar_reviews[:300]}

            Requirements:
            1. Review MUST be 2-4 sentences long and include specific details
            2. Focus on aspects mentioned in similar reviews
            3. Match user's typical style and rating patterns
            4. Keep engagement metrics realistic (most reviews have 0-3 useful votes)
            5. Write review directly after "review:", no line breaks
            6. Base rating mainly on similar reviews and user's typical patterns
            '''

            messages = [{"role": "user", "content": task_description}]
            result = self.reasoning(messages)

            logger.info("\n====== Generated Review ======")
            logger.debug(f"Raw generation result:\n{result}")

            try:
                # Parse result with improved error handling
                lines = result.strip().split('\n')
                parsed_result = {}
                current_field = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for field markers
                    for field in ['stars:', 'useful:', 'funny:', 'cool:', 'review:']:
                        if line.lower().startswith(field):
                            current_field = field[:-1]  # Remove the colon
                            value = line[len(field):].strip()
                            parsed_result[current_field] = value
                            break
                    else:
                        # If no field marker found and we're in review field, append to review
                        if current_field == 'review' and parsed_result.get('review'):
                            parsed_result['review'] += ' ' + line

                # Validate and convert values with more realistic limits
                stars = float(parsed_result.get('stars', '3.0'))
                useful = float(parsed_result.get('useful', '0'))
                funny = float(parsed_result.get('funny', '0'))
                cool = float(parsed_result.get('cool', '0'))
                review_text = parsed_result.get('review', 'No review content provided.').strip()

                # Enforce realistic limits
                stars = max(1.0, min(5.0, stars))
                useful = max(0, min(3, useful))  # More realistic useful limit
                funny = max(0, min(1, funny))  # More realistic funny limit
                cool = max(0, min(2, cool))  # More realistic cool limit

                if len(review_text) > 512:
                    review_text = review_text[:512]

                # Log generated review
                logger.info(f"Stars: {stars}")
                logger.info(f"Useful: {useful}")
                logger.info(f"Funny: {funny}")
                logger.info(f"Cool: {cool}")
                logger.info(f"Review: {review_text}")

                # Print real review for comparison
                if real_reviews:
                    logger.info("\n====== Real Review ======")
                    real_review = real_reviews[0]
                    logger.info(f"Stars: {real_review.get('stars', 0.0)}")
                    logger.info(f"Useful: {real_review.get('useful', 0)}")
                    logger.info(f"Funny: {real_review.get('funny', 0)}")
                    logger.info(f"Cool: {real_review.get('cool', 0)}")
                    logger.info(f"Review: {real_review.get('text', '')}")

                logger.info("\n====== End of Comparison ======\n")
                return stars, useful, funny, cool, review_text

            except Exception as e:
                logger.error(f"Error parsing generated review: {e}")
                logger.error(f"Raw result: {result}")
                return 3.0, 0.0, 0.0, 0.0, "Error processing review."

        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return 3.0, 0.0, 0.0, 0.0, "Error in review generation process."

def main(batch_size: int = 50, max_tasks: int = None):
    """
    运行模拟程序
    Args:
        batch_size: 每批处理的任务数量，用于定期评估
        max_tasks: 最大处理任务数，None表示处理所有任务
    Returns:
        tuple: (all_simulated_data, all_real_data)
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 创建 OpenAI wrapper
        llm = OpenAIWrapper(
            api_key="sk-RdSm0tMmtiBV0ej8hgR3okJUL4yByGpxloq57xMeOPkAPTSJ",
            base_url="https://api.chatanywhere.tech/v1",
            model="gpt-3.5-turbo"
        )

        # 设置数据目录和创建交互工具
        data_directory = r"C:\Users\hq200\Desktop\User Modeling Track\files"
        interaction_tool = CacheInteractionTool(data_dir=data_directory)

        # 创建评估器
        simulation_evaluator = SimulationEvaluator(device="auto")

        # 用于存储所有模拟结果
        all_simulated_data = []
        all_real_data = []

        # 创建任务列表
        tasks = []
        logger.info("正在收集任务...")

        # 获取所有用户和商品ID
        users = interaction_tool.get_all_user_ids()
        items = interaction_tool.get_all_item_ids()

        logger.info(f"发现 {len(users)} 个用户和 {len(items)} 个商品")

        # 优化任务收集逻辑
        task_count = 0
        for user_id in users:
            user_reviews = interaction_tool.get_reviews(user_id=user_id)
            if user_reviews:
                for review in user_reviews:
                    item_id = review['item_id']
                    tasks.append({
                        "user_id": user_id,
                        "item_id": item_id
                    })
                    task_count += 1

                    if max_tasks is not None and task_count >= max_tasks:
                        break

            if max_tasks is not None and task_count >= max_tasks:
                break

        total_tasks = len(tasks)
        logger.info(f"总任务数: {total_tasks}")

        # 批量处理任务
        for i, task in enumerate(tasks, 1):
            try:
                logger.info(f"处理任务 {i}/{total_tasks}")
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

                # 收集模拟数据
                simulated_data = {
                    'stars': stars,
                    'review': review_text,
                    'user_id': task['user_id'],
                    'item_id': task['item_id'],
                    'useful': useful,
                    'funny': funny,
                    'cool': cool
                }
                all_simulated_data.append(simulated_data)

                # 获取真实数据
                real_reviews = interaction_tool.get_reviews(
                    user_id=task['user_id'],
                    item_id=task['item_id']
                )

                if real_reviews:
                    review_data = {
                        'stars': real_reviews[0].get('stars', 0.0),
                        'review': real_reviews[0].get('text', ''),
                        'user_id': task['user_id'],
                        'item_id': task['item_id'],
                        'useful': real_reviews[0].get('useful', 0),
                        'funny': real_reviews[0].get('funny', 0),
                        'cool': real_reviews[0].get('cool', 0)
                    }
                    all_real_data.append(review_data)

                # 定期计算和输出评估指标
                if i % batch_size == 0 or i == total_tasks:
                    metrics = simulation_evaluator.calculate_metrics(
                        simulated_data=all_simulated_data,
                        real_data=all_real_data
                    )
                    logger.info(f"\n中间评估指标 (已完成 {i} 个任务):")
                    logger.info(f"偏好估计: {metrics.preference_estimation:.4f}")
                    logger.info(f"评论生成: {metrics.review_generation:.4f}")
                    logger.info(f"整体质量: {metrics.overall_quality:.4f}\n")

                    # 保存中间结果
                    save_results(all_simulated_data, all_real_data, i)

            except Exception as e:
                logger.error(f"处理任务 {i} 时出错: {str(e)}")
                logger.error("错误详情:", exc_info=True)
                continue

        # 最终评估
        if all_real_data:
            final_metrics = simulation_evaluator.calculate_metrics(
                simulated_data=all_simulated_data,
                real_data=all_real_data
            )

            logger.info("\n最终评估指标:")
            logger.info(f"偏好估计: {final_metrics.preference_estimation:.4f}")
            logger.info(f"评论生成: {final_metrics.review_generation:.4f}")
            logger.info(f"整体质量: {final_metrics.overall_quality:.4f}")

            # 保存最终结果
            save_results(all_simulated_data, all_real_data, 'final')

        return all_simulated_data, all_real_data

    except Exception as e:
        logger.error(f"main 函数执行出错: {str(e)}")
        logger.error("错误详情:", exc_info=True)
        raise


def save_results(simulated_data, real_data, checkpoint):
    """保存模拟结果到文件"""
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存模拟数据
    with open(f"{results_dir}/simulated_data_{checkpoint}.json", 'w', encoding='utf-8') as f:
        json.dump(simulated_data, f, ensure_ascii=False, indent=2)

    # 保存真实数据
    with open(f"{results_dir}/real_data_{checkpoint}.json", 'w', encoding='utf-8') as f:
        json.dump(real_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 可以通过参数控制批量大小和最大任务数
    all_simulated_data, all_real_data = main(batch_size=3, max_tasks=5)

