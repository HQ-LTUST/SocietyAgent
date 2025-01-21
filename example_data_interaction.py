import os
import json

def load_track1_data(dataset_name='amazon'):
    
    """加载track1的数据
    Args:
        dataset_name: 选择数据集 ('amazon', 'yelp', 'goodreads')
    """
    try:
        base_path = r"C:\Users\ASUS\Desktop\AS\AgentSocietyChallenge-main\example\track1"
        dataset_path = os.path.join(base_path, dataset_name.lower())
        
        tasks = {}
        groundtruth = {}
        
        # 加载任务数据
        tasks_path = os.path.join(dataset_path, "tasks")
        for file_name in os.listdir(tasks_path):
            if file_name.endswith('.json'):
                task_id = file_name.split('.')[0]  # 获取task_0等ID
                with open(os.path.join(tasks_path, file_name), 'r', encoding='utf-8') as f:
                    tasks[task_id] = json.load(f)
        
        # 加载groundtruth数据
        groundtruth_path = os.path.join(dataset_path, "groundtruth")
        for file_name in os.listdir(groundtruth_path):
            if file_name.endswith('.json'):
                gt_id = file_name.split('.')[0].replace('groundtruth_', 'task_')
                with open(os.path.join(groundtruth_path, file_name), 'r', encoding='utf-8') as f:
                    groundtruth[gt_id] = json.load(f)
            
        print(f"\n{dataset_name}数据集加载完成:")
        print(f"任务数量: {len(tasks)}")
        print(f"标注数量: {len(groundtruth)}")
        
        return tasks, groundtruth
        
    except Exception as e:
        print(f"加载{dataset_name}数据出错: {e}")
        return None, None

class CacheInteractionTool:
    """交互工具类"""
    def __init__(self, task_data, groundtruth_data=None):
        self.task_data = task_data
        self.groundtruth_data = groundtruth_data
        
    def get_task_info(self):
        """获取任务信息"""
        info = {
            'type': self.task_data['type'],
            'user_id': self.task_data['user_id'],
            'item_id': self.task_data['item_id']
        }
        if self.groundtruth_data:
            info.update({
                'stars': self.groundtruth_data['stars'],
                'review': self.groundtruth_data['review']
            })
        return info

def format_task_data(task_data, groundtruth_data=None):
    """格式化任务数据"""
    formatted = {
        'type': task_data['type'],
        'user_id': task_data['user_id'],
        'item_id': task_data['item_id']
    }
    if groundtruth_data:
        formatted.update({
            'stars': groundtruth_data['stars'],
            'review': groundtruth_data['review']
        })
    return formatted

if __name__ == "__main__":
    # 测试数据加载
    datasets = ['amazon', 'yelp', 'goodreads']
    
    for dataset in datasets:
        print(f"\n测试加载 {dataset} 数据集:")
        tasks, groundtruth = load_track1_data(dataset)
        
        if tasks and groundtruth:
            # 测试第一个样本
            first_task_id = list(tasks.keys())[0]
            print("\n示例数据:")
            print("任务数据:")
            print(format_task_data(tasks[first_task_id]))
            print("\n标注数据:")
            print(format_task_data(tasks[first_task_id], groundtruth[first_task_id]))
            
            # 测试交互工具
            tool = CacheInteractionTool(tasks[first_task_id], groundtruth[first_task_id])
            print("\n交互工具测试:")
            print(tool.get_task_info())
        else:
            print(f"{dataset}数据集加载失败")
