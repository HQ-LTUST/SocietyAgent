from websocietysimulator.tools.example_data_interaction import load_track1_data,format_task_data

if __name__ == "__main__":
    # 测试数据加载
    datasets = ['yelp']  # 选择数据集
    
    for dataset in datasets:
        print(f"\n测试加载 {dataset} 数据集:")
        tasks, groundtruth = load_track1_data(dataset)
        
        if tasks and groundtruth:
            # 提取数据集中的任务内容
            data_list = []
            for task_id, task_data in tasks.items():
                if task_id in groundtruth:
                    groundtruth_data = groundtruth[task_id]
                    formatted_task = format_task_data(task_data, groundtruth_data)
                    data_list.append(formatted_task)
            
            # 输出提取到的数据
            print("\n提取的XX数据:")
            for data in data_list[:5]:   #打印前5个
                print(data)
        else: 
            print(f"{dataset}数据集加载失败")
