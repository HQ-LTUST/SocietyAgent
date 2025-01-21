import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
import numpy as np
from typing import List, Dict, Any
import torch
from tqdm import tqdm
import time
from websocietysimulator.tools.cache_interaction_tool import CacheInteractionTool
from datetime import datetime
import logging

class MemoryModule:
    """记忆模块：存储和管理不同来源的数据"""
    def __init__(self):
        self.user_memory = {}  # 用户历史行为记忆
        self.item_memory = {}  # 商品特征记忆
        self.review_memory = {}  # 评论模式记忆
        self.source_models = {}  # 不同来源的模型
        self.user_preferences = {}  # 用户偏好缓存
        self.item_features = {}     # 商品特征缓存

    def add_user_memory(self, user_id: str, behavior: Dict):
        if user_id not in self.user_memory:
            self.user_memory[user_id] = []
        self.user_memory[user_id].append(behavior)

    def add_item_memory(self, item_id: str, features: Dict):
        self.item_memory[item_id] = features

    def add_review_memory(self, source: str, review_data: Dict):
        if source not in self.review_memory:
            self.review_memory[source] = []
        self.review_memory[source].append(review_data)

    def get_source_specific_data(self, source: str) -> Dict:
        return {
            'users': {k: v for k, v in self.user_memory.items() if v.get('source') == source},
            'items': {k: v for k, v in self.item_memory.items() if v.get('source') == source},
            'reviews': self.review_memory.get(source, [])
        }

    def update_user_preference(self, user_id, item_id, rating):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'liked': [], 'disliked': []}
        
        if rating >= 4:
            self.user_preferences[user_id]['liked'].append(item_id)
        elif rating <= 2:
            self.user_preferences[user_id]['disliked'].append(item_id)

class PlanningModule:
    """规划模块：制定推荐策略"""
    def __init__(self, llm):
        self.llm = llm

    def create_recommendation_plan(self, user_id: str, memory_module: MemoryModule) -> List[Dict]:
        user_history = memory_module.user_memory.get(user_id, [])
        
        plan_prompt = f"""基于用户 {user_id} 的历史行为，制定推荐计划：
        1. 分析用户历史偏好
        2. 确定商品匹配策略
        3. 生成推荐候选列表
        用户历史: {json.dumps(user_history, ensure_ascii=False)}
        """
        
        response = self.llm.generate(plan_prompt)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> List[Dict]:
        # 解析LLM返回的计划
        try:
            return json.loads(response)
        except:
            return []

class ReasoningModule:
    """推理模块：执行推荐决策"""
    def __init__(self, llm):
        self.llm = llm

    def reason_recommendation(self, 
                            user_id: str, 
                            candidate_items: List[str],
                            memory_module: MemoryModule) -> str:
        user_history = memory_module.user_memory.get(user_id, [])
        items_info = {item_id: memory_module.item_memory.get(item_id, {}) 
                     for item_id in candidate_items}
        
        reasoning_prompt = f"""基于以下信息为用户 {user_id} 选择最佳商品：
        用户历史: {json.dumps(user_history, ensure_ascii=False)}
        候选商品: {json.dumps(items_info, ensure_ascii=False)}
        
        请直接返回最合适的商品ID。
        """
        
        return self.llm.generate(reasoning_prompt).strip()

class TrainingMonitor:
    """训练监控模块：监控和报告训练进度"""
    def __init__(self):
        self.metrics = {
            'total_samples': 0,
            'processed_samples': 0,
            'current_batch': 0,
            'total_batches': 0,
            'source_metrics': {},
            'current_source': '',
            'training_history': []
        }
        self.start_time = None

    def start_training(self, total_samples: int, batch_size: int, source: str):
        """开始训练监控"""
        self.start_time = time.time()
        self.metrics['total_samples'] = total_samples
        self.metrics['total_batches'] = total_samples // batch_size
        self.metrics['current_source'] = source
        print(f"\n开始训练 {source} 数据源:")
        print(f"总样本数: {total_samples}")
        print(f"总批次数: {self.metrics['total_batches']}")
        print(f"批次大小: {batch_size}")

    def update_batch_progress(self, batch_idx: int, batch_metrics: Dict, batch_size: int):
        """更新批次进度
        Args:
            batch_idx: 当前批次索引
            batch_metrics: 当前批次的训练指标
            batch_size: 批次大小
        """
        current_batch = batch_idx + 1
        self.metrics['current_batch'] = current_batch
        
        # 计算实际处理的样本数
        processed_samples = min(
            current_batch * batch_size,  # 理论处理数量
            self.metrics['total_samples']  # 总样本数上限
        )
        self.metrics['processed_samples'] = processed_samples
        
        # 计算实际进度
        progress = processed_samples / self.metrics['total_samples']
        
        # 计算时间
        elapsed_time = time.time() - self.start_time
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time
        
        # 更新源数据指标
        self.metrics['source_metrics'][self.metrics['current_source']] = batch_metrics
        
        # 打印详细进度报告
        print(f"\r进度: {processed_samples}/{self.metrics['total_samples']} 样本"
              f"({progress:.1%}) | "
              f"批次: {current_batch}/{self.metrics['total_batches']} | "
              f"已用时间: {elapsed_time:.1f}s | "
              f"预计剩余: {remaining_time:.1f}s | "
              f"损失: {batch_metrics.get('loss', 'N/A'):.4f} | "
              f"准确率: {batch_metrics.get('accuracy', 'N/A'):.2%}", 
              end='')

    def log_epoch_metrics(self, epoch_metrics: Dict):
        """记录每轮训练的指标"""
        self.metrics['training_history'].append(epoch_metrics)
        print(f"\n\n{self.metrics['current_source']} 训练轮次完成:")
        for metric_name, value in epoch_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def get_training_summary(self) -> Dict:
        """获取训练总结"""
        return {
            'total_time': time.time() - self.start_time,
            'final_metrics': self.metrics['source_metrics'],
            'training_history': self.metrics['training_history']
        }

class BatchTrainer:
    """批量训练管理器"""
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.monitor = TrainingMonitor()

    def train_source_model(self, source_data: Dict, memory_module: MemoryModule, source: str):
        """针对特定来源的数据进行批量训练"""
        total_samples = len(source_data['reviews'])
        self.monitor.start_training(total_samples, self.batch_size, source)
        
        # 计算实际的总批次数（考虑不能整除的情况）
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        epoch_metrics = {
            'total_loss': 0,
            'total_accuracy': 0,
            'processed_batches': 0
        }
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, total_samples)  # 防止越界
            
            # 获取当前批次实际大小
            current_batch_size = batch_end - batch_start
            
            batch_data = {
                'reviews': source_data['reviews'][batch_start:batch_end],
                'users': {k: v for k, v in source_data['users'].items() 
                         if any(r['user_id'] == k for r in source_data['reviews'][batch_start:batch_end])},
                'items': {k: v for k, v in source_data['items'].items() 
                         if any(r['item_id'] == k for r in source_data['reviews'][batch_start:batch_end])}
            }
            
            # 训练批次并获取指标
            batch_metrics = self._train_batch(batch_data, memory_module, source)
            
            # 更新进度和指标（传入实际的batch_size）
            self.monitor.update_batch_progress(
                batch_idx, 
                batch_metrics,
                current_batch_size
            )
            
            # 根据实际处理的样本数更新累积指标
            epoch_metrics['total_loss'] += batch_metrics.get('loss', 0) * current_batch_size
            epoch_metrics['total_accuracy'] += batch_metrics.get('accuracy', 0) * current_batch_size
            epoch_metrics['processed_batches'] += current_batch_size

        # 计算加权平均的轮次指标
        avg_epoch_metrics = {
            'avg_loss': epoch_metrics['total_loss'] / total_samples,
            'avg_accuracy': epoch_metrics['total_accuracy'] / total_samples
        }
        self.monitor.log_epoch_metrics(avg_epoch_metrics)
        
        return self.monitor.get_training_summary()

    def _train_batch(self, 
                    batch_data: Dict, 
                    memory_module: MemoryModule,
                    source: str):
        """训练单个批次"""
        # 这里实现具体的训练逻辑
        pass

class MyRecommendationAgent:
    def __init__(self, llm, data_dir="C:/Users/13752/Desktop/AgentSocietyChallenge-main/digital"):
        """初始化推荐代理
        Args:
            llm: LLM实例
            data_dir: 数据目录路径
        """
        self.llm = llm
        # 只初始化一次缓存工具
        self.cache_tool = CacheInteractionTool(data_dir)
        print(f"推荐代理初始化完成，使用数据目录: {data_dir}")
        self.memory = MemoryModule()
        self.planning = PlanningModule(llm)
        self.reasoning = ReasoningModule(llm)
        self.trainer = BatchTrainer()
        self.training_summaries = {}
        self.setup_logging()

    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rec_agent.log'),
                logging.StreamHandler()
            ]
        )

    def train_with_digital_data(self, digital_data: Dict):
        """使用digital数据训练模型"""
        print("\n开始训练数字数据集...")
        
        for source in digital_data['sources']:
            print(f"\n处理数据源: {source}")
            source_data = self.memory.get_source_specific_data(source)
            
            # 训练并获取摘要
            summary = self.trainer.train_source_model(source_data, self.memory, source)
            self.training_summaries[source] = summary
            
            print(f"\n{source} 训练完成:")
            print(f"总训练时间: {summary['total_time']:.2f}秒")
            print(f"最终准确率: {summary['final_metrics'][source].get('accuracy', 'N/A'):.2%}")

    def workflow(self) -> List[str]:
        """推荐工作流程"""
        try:
            # 1. 获取并预处理用户信息
            user_id = self.task['user_id']
            user_info = self.cache_tool.get_user(user_id)
            
            # 2. 获取候选商品信息
            candidate_items = []
            for item_id in self.task['candidate_list']:
                item_info = self.cache_tool.get_item(item_id)
                if item_info:
                    candidate_items.append(item_info)
            
            # 3. 获取用户历史评论
            user_reviews = self.cache_tool.get_reviews(user_id)
            
            # 4. 构建推荐提示词
            prompt = f"""作为推荐系统专家，请基于以下信息选择最适合的商品：
            用户信息：{json.dumps(user_info, ensure_ascii=False)}
            用户历史评论：{json.dumps(user_reviews, ensure_ascii=False)[:1000]}  # 限制长度
            候选商品：{json.dumps(candidate_items, ensure_ascii=False)}
            
            请直接返回一个商品ID。"""
            
            # 5. 使用推理模块获取推荐
            result = self.reasoning(prompt)
            result = result.strip().replace('[', '').replace(']', '').strip('"').strip("'")
            
            return [result] if result in self.task['candidate_list'] else [self.task['candidate_list'][0]]
            
        except Exception as e:
            print(f"推荐过程出错: {e}")
            return [self.task['candidate_list'][0]]

    def get_user_history(self, user_id):
        # 获取用户历史数据
        return self.cache_tool.get_reviews(user_id)

    def get_item_info(self, item_id):
        # 获取商品信息
        return self.cache_tool.get_item(item_id)

    def train_on_cached_data(self, batch_size=1000, epochs=1, max_samples=None):
        try:
            print("\n=== 开始训练缓存数据 ===")
            start_time = datetime.now()
            
            # 获取评论数据总量并存储到列表中
            reviews_list = []
            with self.cache_tool.review_env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key.startswith(b'user_') or key.startswith(b'item_'):
                        continue
                    review = json.loads(value.decode())
                    reviews_list.append(review)
                    if max_samples and len(reviews_list) >= max_samples:
                        break
            
            total_reviews = len(reviews_list)
            print(f"加载评论数: {total_reviews:,}")
            
            processed = 0
            for epoch in range(epochs):
                print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
                epoch_start = datetime.now()
                
                # 使用tqdm创建进度条，直接遍历reviews_list
                with tqdm(total=total_reviews, desc=f"Epoch {epoch + 1}", 
                         unit="reviews", ncols=100) as pbar:
                    
                    batch_data = []
                    for review in reviews_list:
                        # 获取用户和商品信息
                        user_data = self.cache_tool.get_user(review['user_id'])
                        item_data = self.cache_tool.get_item(review['item_id'])
                        
                        if user_data and item_data:
                            # 构建训练样本
                            training_sample = {
                                # 用户特征
                                'user_id': review['user_id'],
                                'user_avg_stars': user_data.get('average_stars', 0),
                                'user_review_count': user_data.get('review_count', 0),
                                
                                # 商品特征
                                'item_id': review['item_id'],
                                'item_avg_rating': item_data.get('average_rating', 0),
                                'item_categories': item_data.get('categories', []),
                                'item_price': item_data.get('price', 0),
                                
                                # 评论特征
                                'rating': review.get('stars', 0),
                                'text': review.get('text', ''),
                                'helpful_votes': review.get('helpful_vote', 0)
                            }
                            
                            batch_data.append(training_sample)
                            
                            # 当达到batch_size时进行训练
                            if len(batch_data) >= batch_size:
                                if self._train_batch(batch_data):
                                    processed += len(batch_data)
                                    pbar.update(len(batch_data))
                                    
                                    # 显示额外信息
                                    elapsed = (datetime.now() - start_time).total_seconds()
                                    rate = processed / elapsed if elapsed > 0 else 0
                                    pbar.set_postfix({
                                        'speed': f'{rate:.1f} reviews/s',
                                        'elapsed': f'{elapsed/60:.1f}min'
                                    })
                                else:
                                    print("\n批次处理失败，跳过")
                                
                                batch_data = []
                    
                    # 处理最后一个不完整的batch
                    if batch_data:
                        if self._train_batch(batch_data):
                            processed += len(batch_data)
                            pbar.update(len(batch_data))
                
                epoch_time = (datetime.now() - epoch_start).total_seconds()
                print(f"\nEpoch {epoch + 1} 完成! 用时: {epoch_time/60:.1f}分钟")
            
            total_time = (datetime.now() - start_time).total_seconds()
            print("\n=== 训练完成 ===")
            print(f"总用时: {total_time/60:.1f}分钟")
            print(f"处理数据量: {processed:,}")
            
            return {
                'status': 'success',
                'total_processed': processed,
                'epochs': epochs,
                'total_time_minutes': total_time/60
            }
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _train_batch(self, batch_data):
        """训练一个批次的数据"""
        try:
            # 1. 按用户分组处理数据
            user_groups = {}
            for sample in batch_data:
                user_id = sample['user_id']
                if user_id not in user_groups:
                    user_groups[user_id] = []
                user_groups[user_id].append(sample)
            
            # 2. 对每个用户进行偏好学习
            for user_id, user_samples in user_groups.items():
                # 构建用户偏好学习提示
                positive_samples = [s for s in user_samples if s['rating'] >= 4]
                negative_samples = [s for s in user_samples if s['rating'] <= 2]
                
                if not positive_samples and not negative_samples:
                    continue
                    
                prompt = f"""作为用户行为分析师，请分析用户 {user_id} 的偏好:

正面评价的商品:
{self._format_samples(positive_samples)}

负面评价的商品:
{self._format_samples(negative_samples)}

请总结:
1. 用户喜欢的商品特征
2. 用户不喜欢的商品特征
3. 用户的价格敏感度
4. 用户关注的商品类别
"""
                
                # 使用LLM分析用户偏好
                try:
                    # 使用正确的 generate 方法
                    response = self.llm.generate(prompt)
                    
                    # 存储用户偏好分析结果
                    if response:
                        self._update_user_preferences(user_id, response)
                        
                except Exception as e:
                    print(f"LLM处理出错 ({user_id}): {e}")
                    continue
                
                # 3. 更新商品特征理解
                for sample in user_samples:
                    item_id = sample['item_id']
                    self._update_item_features(item_id, sample)
                    
            return True
            
        except Exception as e:
            print(f"批次训练出错: {e}")
            return False

    def _format_samples(self, samples):
        """格式化样本数据用于提示"""
        if not samples:
            return "无"
        
        formatted = []
        for s in samples:
            item_info = f"""- 商品ID: {s['item_id']}
      类别: {', '.join(s['item_categories'][:3]) if s['item_categories'] else '未知'}
      价格: {s['item_price']}
      评分: {s['rating']}
      评论: {s['text'][:100] + '...' if len(s['text']) > 100 else s['text']}"""
            formatted.append(item_info)
        
        return "\n".join(formatted)

    def _update_user_preferences(self, user_id: str, analysis: str):
        """更新用户偏好信息"""
        try:
            with self.cache_tool.user_env.begin(write=True) as txn:
                user_key = user_id.encode()
                user_data = json.loads(txn.get(user_key).decode())
                
                # 添加或更新偏好分析
                if 'preference_analysis' not in user_data:
                    user_data['preference_analysis'] = []
                user_data['preference_analysis'].append({
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis
                })
                
                # 保留最新的5次分析
                user_data['preference_analysis'] = user_data['preference_analysis'][-5:]
                
                txn.put(user_key, json.dumps(user_data).encode())
                
        except Exception as e:
            print(f"更新用户偏好时出错 ({user_id}): {e}")

    def _update_item_features(self, item_id: str, sample: dict):
        """更新商品特征理解"""
        try:
            with self.cache_tool.item_env.begin(write=True) as txn:
                item_key = item_id.encode()
                item_data = json.loads(txn.get(item_key).decode())
                
                # 确保评分是数值类型
                rating = float(sample['rating'])
                
                # 更新评分统计
                if 'rating_stats' not in item_data:
                    item_data['rating_stats'] = {
                        'count': 0,
                        'sum': 0.0,
                        'distribution': {str(i): 0 for i in range(1, 6)}
                    }
                
                stats = item_data['rating_stats']
                stats['count'] += 1
                stats['sum'] += rating
                stats['distribution'][str(int(rating))] += 1
                
                # 更新平均评分
                item_data['average_rating'] = stats['sum'] / stats['count']
                
                # 保存更新后的数据
                txn.put(item_key, json.dumps(item_data).encode())
                
        except Exception as e:
            print(f"更新商品特征时出错 ({item_id}): {str(e)}")
            print(f"样本数据: {json.dumps(sample, ensure_ascii=False)}")

class RecReasoning:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, task_description: str):
        try:
            reasoning_result = self.llm.generate(str(task_description))
            return reasoning_result if reasoning_result else ""
        except Exception as e:
            print(f"推理过程出错: {e}")
            return ""