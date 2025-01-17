import unittest
from unittest.mock import MagicMock
from websocietysimulator.agent.simulation_agent import SimulationAgent

class TestSimulationAgent(unittest.TestCase):
    def setUp(self):
        # 创建模拟对象
        self.llm = MagicMock()
        self.agent = SimulationAgent(self.llm)
        
        # 自建模拟任务，纯测试工作流
        mock_task = MagicMock()
        mock_task.to_dict.return_value = {
            'user_id': 'test_user_1',
            'item_id': 'test_item_1'
        }
        self.agent.insert_task(mock_task)

        # 模拟 interaction_tool（模拟一下调用功能
        self.agent.interaction_tool = MagicMock()
        
        # 设置用户数据
        self.agent.interaction_tool.get_user.return_value = {
            'user_id': 'test_user_1',
            'name': 'Test User',
            'review_count': 10
        }
        
        # 设置商家数据
        self.agent.interaction_tool.get_item.return_value = {
            'item_id': 'test_item_1',
            'name': 'Test Business',
            'stars': 4.0
        }
        
        # 设置评论数据
        self.agent.interaction_tool.get_reviews.return_value = [{
            'review_id': 'test_review_1',
            'text': 'Great service and atmosphere!',
            'stars': 5,
            'user_id': 'test_user_1',
            'item_id': 'test_item_1'
        }]

        # 模拟 planning 和 reasoning
        self.agent.planning = MagicMock(return_value="Generated plan")
        # 确保 reasoning 返回正确格式的字符串
        self.agent.reasoning = MagicMock(
            return_value="""stars: 4.5
review: This is a great business with excellent service."""
        )

    def test_analyze_user_history(self):
        """测试用户历史分析"""
        result = self.agent.analyze_user_history()
        self.assertIn('user_profile', result)
        self.assertIn('review_history', result)
        self.assertEqual(result['user_profile']['user_id'], 'test_user_1')

    def test_analyze_context(self):
        """测试情境分析"""
        result = self.agent.analyze_context()
        self.assertIn('business_profile', result)
        self.assertIn('business_reviews', result)
        self.assertEqual(result['business_profile']['item_id'], 'test_item_1')

    def test_analyze_sentiment(self):
        """测试情感分析"""
        result = self.agent.analyze_sentiment()
        self.assertIn('sentiment_scores', result)
        self.assertIn('review_texts', result)
        self.assertEqual(result['sentiment_scores'][0], 5)

    def test_consider_memory(self):
        """测试记忆模块"""
        result = self.agent.consider_memory()
        self.assertIn('recent_activity', result)
        self.assertIn('patterns', result)
        self.assertTrue(isinstance(result['patterns'], dict))

    def test_generate_final_review(self):
        """测试最终评论生成"""
        user_history = self.agent.analyze_user_history()
        context_info = self.agent.analyze_context()
        sentiment = self.agent.analyze_sentiment()
        reasoning_result = """stars: 4.5
review: This is a great business with excellent service."""
        memory_impact = self.agent.consider_memory()

        result = self.agent.generate_final_review(
            user_history,
            context_info,
            sentiment,
            reasoning_result,
            memory_impact
        )

        self.assertIn('stars', result)
        self.assertIn('review', result)
        self.assertEqual(result['stars'], 4.5)

    def test_workflow_with_mocked_data(self):
        """测试完整工作流程"""
        result = self.agent.workflow()
        
        # 打印调试信息
        print("Workflow result:", result)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn('stars', result)
        self.assertIn('review', result)
        self.assertEqual(result['stars'], 4.5)
        self.assertEqual(
            result['review'],
            "This is a great business with excellent service."
        )

    def test_workflow_error_handling(self):
        """测试错误处理"""
        # 保存原始的 reasoning mock
        original_reasoning = self.agent.reasoning
        
        try:
            # 测试不同类型的错误情况
            error_cases = [
                (Exception("Test error"), "An error occurred during the simulation: Test error"),
                (ValueError("Invalid input"), "An error occurred during the simulation: Invalid input"),
                (None, "An error occurred during the simulation: Reasoning failed to generate result")
            ]
            
            for error, expected_message in error_cases:
                # 设置模拟错误
                if error is None:
                    self.agent.reasoning = MagicMock(return_value=None)
                else:
                    self.agent.reasoning = MagicMock(side_effect=error)
                
                # 执行工作流
                result = self.agent.workflow()
                
                # 验证错误处理结果
                self.assertEqual(result['stars'], 0)
                self.assertEqual(result['review'], expected_message)
                
        finally:
            # 恢复原始的 reasoning mock
            self.agent.reasoning = original_reasoning

    def test_insert_task(self):
        """测试任务插入"""
        mock_task = MagicMock()
        mock_task.to_dict.return_value = {
            'user_id': 'test_user_1',
            'item_id': 'test_item_1'
        }
        self.agent.insert_task(mock_task)
        self.assertEqual(self.agent.task, mock_task.to_dict.return_value)

    def test_reasoning(self):
        """测试推理功能"""
        task_description = "Test task description"
        result = self.agent.reasoning(task_description)
        
        # 验证结果格式
        self.assertIsNotNone(result)
        self.assertIn('stars:', result.lower())
        self.assertIn('review:', result.lower())
        
        # 验证内容解析
        stars_line = next((line for line in result.split('\n') 
                          if 'stars:' in line.lower()), None)
        review_line = next((line for line in result.split('\n') 
                           if 'review:' in line.lower()), None)
                           
        self.assertIsNotNone(stars_line)
        self.assertIsNotNone(review_line)
        
        # 验证评分范围
        stars = float(stars_line.split(':')[1].strip())
        self.assertTrue(1 <= stars <= 5)

    def test_workflow_success(self):
        """测试成功的工作流程"""
        # 设置预期的返回值
        expected_review = "This is a great business with excellent service."
        expected_stars = 4.5
        
        # 执行工作流
        result = self.agent.workflow()
        
        # 打印详细信息用于调试
        print("Workflow result:", result)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn('stars', result)
        self.assertIn('review', result)
        self.assertEqual(result['stars'], expected_stars)
        self.assertEqual(result['review'], expected_review)

if __name__ == '__main__':
    unittest.main()

    # 已测试完，workflow2.0版可正常运行
    # 完整工作流程成功执行，生成了预期的评论和评分，所有测试都按预期执行，包括：正常功能测试，错误处理测试，边界条件测试，完整工作流程测试




