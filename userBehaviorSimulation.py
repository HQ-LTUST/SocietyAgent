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
    """Participant's implementation of SimulationAgent."""

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

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))

            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')

            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')

            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            useful: [count]
            funny: [count] 
            cool: [count]
            review: [your review]
            '''

            result = self.reasoning(task_description)

            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                useful_line = [line for line in result.split('\n') if 'useful:' in line][0]
                funny_line = [line for line in result.split('\n') if 'funny:' in line][0]
                cool_line = [line for line in result.split('\n') if 'cool:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            useful = float(useful_line.split(':')[1].strip())
            funny = float(funny_line.split(':')[1].strip())
            cool = float(cool_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]

            return stars, useful, funny, cool, review_text
        except Exception as e:
            print(f"Error in workflow: {e}")
            return 0.0, 0.0, 0.0, 0.0, ""




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
    all_simulated_data, all_real_data = main(batch_size=50, max_tasks=1)

