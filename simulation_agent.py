import sys
print("Current working directory:", sys.path)
print("Loaded modules:", sys.modules)

from typing import Dict, Any
from .agent import Agent

from ..llm import LLMBase
import json

class SimulationAgent(Agent):
    def __init__(self, llm):
        super().__init__(llm)
        self.task = None
        self.interaction_tool = None

    def insert_task(self, task):
        """Insert task into agent"""
        self.task = task.to_dict()
        
    def analyze_user_history(self):
        """分析用户历史数据"""
        user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
        user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
        return {
            'user_profile': user_info,
            'review_history': user_reviews
        }
        
    def analyze_context(self):
        """分析情境信息"""
        business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
        business_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
        return {
            'business_profile': business_info,
            'business_reviews': business_reviews
        }
        
    def analyze_sentiment(self):
        """情感分析"""
        reviews = self.interaction_tool.get_reviews(
            user_id=self.task['user_id'],
            item_id=self.task['item_id']
        )
        return {
            'sentiment_scores': [review['stars'] for review in reviews],
            'review_texts': [review['text'] for review in reviews]
        }

    def planning(self, task_description: dict) -> str:
        """Generate plan based on task description"""
        plan = self.llm.generate(task_description)
        return plan
        
    def reasoning(self, task_description: str) -> str:
        """Reasoning based on task description"""
        try:
            # 确保 llm 存在且可用
            if not self.llm:
                raise ValueError("LLM not initialized")
            
            # 添加更明确的提示
            prompt = f"""
            Based on the following information, generate a review and rating:
            {task_description}
            
            Please ensure your response follows this exact format:
            stars: [rating between 1-5]
            review: [detailed review text]
            """
            
            result = self.llm.generate(prompt)
            
            # 验证结果格式
            if not result or not isinstance(result, str):
                raise ValueError("Invalid reasoning result format")
            
            # 验证结果是否包含必要的部分
            if 'stars:' not in result.lower() or 'review:' not in result.lower():
                raise ValueError("Missing required fields in reasoning result")
            
            return result
            
        except Exception as e:
            print(f"Error in reasoning: {e}")
            # 返回一个有效的默认值而不是 None
            return """stars: 3.0
review: This is a default review generated due to an error in the reasoning process."""
        
    def consider_memory(self):
        """考虑历史影响"""
        recent_reviews = self.interaction_tool.get_reviews(
            user_id=self.task['user_id'],
            limit=5  # 最近5条评论
        )
        return {
            'recent_activity': recent_reviews,
            'patterns': self._analyze_patterns(recent_reviews)
        }
    
    def _analyze_patterns(self, reviews):
        """分析评论模式"""
        if not reviews:
            return {}
        return {
            'avg_rating': sum(review['stars'] for review in reviews) / len(reviews),
            'common_themes': self._extract_themes(reviews)
        }
    
    def _extract_themes(self, reviews):
        """提取评论主题"""
        # 这里可以使用更复杂的主题提取算法
        return [review['text'][:100] for review in reviews]
        
    def generate_final_review(self, user_history, context_info, 
                            sentiment, reasoning_result, memory_impact):
        """整合信息生成最终评论"""
        try:
            # 解析推理结果
            stars_line = next((line for line in reasoning_result.split('\n') 
                             if 'stars:' in line.lower()), None)
            review_line = next((line for line in reasoning_result.split('\n') 
                              if 'review:' in line.lower()), None)

            if not stars_line or not review_line:
                raise ValueError("Invalid reasoning result format")

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            # 验证结果
            if not (1 <= stars <= 5):
                raise ValueError(f"Invalid star rating: {stars}")
            if len(review_text) > 512:
                review_text = review_text[:509] + "..."

            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            print(f"Error in generate_final_review: {e}")
            return {
                "stars": 0,
                "review": f"Error generating review: {str(e)}"
            }

    def workflow(self) -> Dict[str, Any]:
        """主工作流程"""
        try:
            if not self.task:
                raise ValueError("No task has been inserted.")

            print("Starting workflow...")  # 添加调试信息

            # 1. 数据收集和分析
            user_history = self.analyze_user_history()
            print("User history analyzed:", bool(user_history))  # 添加调试信息
            
            context_info = self.analyze_context()
            print("Context analyzed:", bool(context_info))  # 添加调试信息
            
            sentiment = self.analyze_sentiment()
            print("Sentiment analyzed:", bool(sentiment))  # 添加调试信息
            
            # 2. 计划生成
            plan = self.planning(task_description=self.task)
            print("Plan generated:", bool(plan))  # 添加调试信息
            
            # 3. 推理和分析
            task_description = f'''
            Task: Write a review as a Yelp user.
            
            User Profile:
            {json.dumps(user_history['user_profile'], ensure_ascii=False, indent=2)}
            
            Previous Reviews (up to 3):
            {json.dumps(user_history['review_history'][:3], ensure_ascii=False, indent=2)}
            
            Business Information:
            {json.dumps(context_info['business_profile'], ensure_ascii=False, indent=2)}
            
            Recent Business Reviews (up to 3):
            {json.dumps(context_info['business_reviews'][:3], ensure_ascii=False, indent=2)}
            
            Sentiment Analysis:
            {json.dumps(sentiment, ensure_ascii=False, indent=2)}
            
            Please provide your rating and review in the following format:
            stars: [1-5]
            review: [your detailed review]
            '''
            
            print("Generating reasoning...")  # 添加调试信息
            reasoning_result = self.reasoning(task_description)
            print("Reasoning result received:", bool(reasoning_result))  # 添加调试信息

            if not reasoning_result:
                raise ValueError("Reasoning failed to generate result")

            # 4. 考虑历史影响
            memory_impact = self.consider_memory()
            print("Memory impact considered:", bool(memory_impact))  # 添加调试信息

            # 5. 生成最终评论
            final_review = self.generate_final_review(
                user_history,
                context_info,
                sentiment,
                reasoning_result,
                memory_impact
            )
            print("Final review generated:", bool(final_review))  # 添加调试信息
            
            return final_review

        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            return {
                "stars": 0,
                "review": f"An error occurred during the simulation: {str(e)}"
            }