import argparse
import json
import os
import random
import pandas as pd
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Model
from utils.meters import AverageMeter
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics.pairwise import cosine_similarity  # 新增导入

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


import json
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import os


class RouterDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        size: Optional[int] = None,
        data_type: str = "multi_attempt",
        dataset_id: int = 0,
        window_size: int = 3,  # 滑动窗口大小
        similarity_threshold: float = 0.95,  # 相似度阈值
        embedding_model_name: str = 'microsoft/mdeberta-v3-base',
        embedding_cache_path: str = 'embeddings.pt',
    ):
        """
        A PyTorch Dataset for routing questions with similarity-based sliding window support using DeBERTa-v3.

        Args:
            data_path (str): Path to the JSON data file.
            source_max_token_len (int, optional): Maximum token length for the source. Defaults to 512.
            target_max_token_len (int, optional): Maximum token length for the target. Defaults to 512.
            size (Optional[int], optional): Desired size of the dataset. If provided, the data is repeated to reach this size. Defaults to None.
            data_type (str, optional): Type of data. Defaults to "multi_attempt".
            dataset_id (int, optional): Identifier for the dataset. Defaults to 0.
            window_size (int, optional): Sliding window size. Defaults to 3.
            similarity_threshold (float, optional): Threshold for similarity to consider as the same dialogue round. Defaults to 0.7.
            embedding_model_name (str, optional): Hugging Face model name for embedding. Defaults to 'microsoft/mdeberta-v3-base'.
            device (str, optional): Device to run the embedding model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            embedding_cache_path (str, optional): Path to cache embeddings. Defaults to 'embeddings.pt'.
        """
        self.data_path = data_path
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.size = size
        self.data_type = data_type
        self.dataset_id = dataset_id
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.embedding_cache_path = embedding_cache_path

        # 加载数据
        self.data = self.load_data(self.data_path)

        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name,
            truncation_side='left',
            padding=True
        )

        # 初始化嵌入模型
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
        self.embedding_model.eval()  # 设置为评估模式
        self.router_node=list(self.data[0]['scores'].keys())
        # 计算或加载嵌入
        self.embeddings = self.load_or_compute_embeddings()

        # 预处理数据为相似度划分的滑动窗口形式
        self.processed_data = self.create_similarity_based_windows(
            self.data, self.window_size, self.similarity_threshold
        )

    def load_data(self, data_path) :
        """
        加载JSON格式的数据文件。

        Args:
            data_path (str): 数据文件路径。

        Returns:
            List[Dict[str, Any]]: 加载的数据列表。
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSON from {data_path}: {e}")
            else:
                raise ValueError("Unsupported file format. Only JSON files are supported.")
        return data
    def load_or_compute_embeddings(self):
        """
        加载嵌入缓存，如果不存在则计算嵌入并缓存。
        """
        if os.path.exists(self.embedding_cache_path):
            print(f"Loading embeddings from {self.embedding_cache_path}")
            embeddings = torch.load(self.embedding_cache_path)
            if len(embeddings) != len(self.data):
                print("Embedding size mismatch. Recomputing embeddings.")
                embeddings = self.compute_embeddings([entry['question'] for entry in self.data],64)
                torch.save(embeddings, self.embedding_cache_path)
        else:
            print("Computing embeddings...")
            embeddings = self.compute_embeddings([entry['question'] for entry in self.data])
            torch.save(embeddings, self.embedding_cache_path)
            print(f"Embeddings saved to {self.embedding_cache_path}")
        return embeddings

    def compute_embeddings(self, texts, batch_size) :
        """
        批量计算文本的嵌入。
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
                batch_texts = texts[i:i + batch_size]
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.source_max_token_len,
                    return_tensors='pt'
                ).to(self.device)
                model_output = self.embedding_model(**encoded_input)
                # 平均池化所有 token 的隐藏状态
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                # 归一化嵌入以便于余弦相似度计算
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.extend(batch_embeddings.cpu())
        return embeddings

    def create_similarity_based_windows(
        self, data, window_size, threshold
    ) :
        """
        根据问题的相似度将数据划分为不同的对话轮次，并生成滑动窗口。
        Args:
            data (List[Dict[str, Any]]): 原始数据列表。
            window_size (int): 滑动窗口大小。
            threshold (float): 相似度阈值。
        """
        processed = []
        current_round = []
        previous_embedding = None

        for i, entry in enumerate(data):
            current_embedding = self.embeddings[i]

            if previous_embedding is not None:
                # 计算与前一个问题的余弦相似度
                cosine_sim = F.cosine_similarity(current_embedding, previous_embedding, dim=0).item()
                if cosine_sim >= threshold:
                    current_round.append(entry)
                else:
                    # 对当前轮次生成滑动窗口
                    # 开始新的轮次
                    current_round = [entry]
            else:
                current_round.append(entry)

            previous_embedding = current_embedding
            processed.append(self.generate_windows_from_round(current_round,window_size=10000))
        return processed

    def generate_windows_from_round(
        self, round_data, window_size
    ) :
        """
        从一个对话轮次中生成滑动窗口。
        """

        combined_question = " ".join([item['question'] for item in round_data])
        combined_scores = round_data[-1]['scores']  # 使用最后一个问题的 scores
        cluster_id =round_data[-1].get('cluster_id', 0)
        windows={
                'question': combined_question,
                'scores': combined_scores,
                'cluster_id': cluster_id
            }
        return windows

    def __getitem__(self, index):
        data_point = self.processed_data[index]
        scores = torch.tensor(list(data_point['scores'].values()))
        question = data_point['question']
        question_id = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()
        cluster_id = data_point['cluster_id'] if "cluster_id" in data_point else 0
        return question_id, scores, self.dataset_id, cluster_id
    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.processed_data)

    def register_tokenizer(self, tokenizer: Any):
        """
        Registers a tokenizer for the dataset.

        Args:
            tokenizer (Any): Tokenizer instance.
        """
        self.tokenizer = tokenizer



# using inner product first
class RouterModule(nn.Module):
    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function = "cos"):
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)
        self.similarity_function = similarity_function
        # 初始化嵌入缓存
        self.embedding_cache = {}

    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T


    '''The forward function pass the input to Router and compute the similarity between model output and trainable embedding'''
    def forward(self, t=1, **input_kwargs):
        x = self.backbone(**input_kwargs)
        # We used the first token as classifier token.
        hidden_state = x['last_hidden_state'][:,0,:]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False,dim=-1)

        for i in range(top_k):
            positive_index = top_index[:,i].view(-1,1)

            # If positive model does not well, skip this.
            mask = torch.where(top_index_true[:,i].view(-1,1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            # make the last_x ignore the true items
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            temp_x = torch.concat([top_x, last_x], dim=-1)

            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:,0])
            log_x = log_x * mask 
            # * mask2
            loss += torch.mean(-log_x)
        return loss
    
    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for dataset_id in dataset_ids:
            positive_indexs = torch.nonzero(dataset_ids == dataset_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. dataset_id:", dataset_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].squeeze()
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. cluster_id:", cluster_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].view(-1)
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss


# evaluation the router with dataset. 
def evaluation(router_model, dataset_paths, dataset_types, tokenizer, batch_size, device,similarity):    
    result = {}
    with torch.no_grad():
        assert len(dataset_paths) == len(dataset_types)
        for index, data_path in enumerate(dataset_paths):
            test_dataset = RouterDataset(data_path=data_path,window_size=3,similarity_threshold=similarity)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            correct_predict = 0
            correct = 0
            for batch in data_loader:
                inputs, scores, _, _ = batch
                inputs = inputs.to(device)
                scores = scores.to(device)
                x, _ = router_model.forward(**inputs)
                softmax_x = nn.Softmax(dim=1)(x)
                _, max_index = torch.max(softmax_x, dim=1)

                _, target_max_index = torch.max(scores, dim=1)
                equals = max_index.eq(target_max_index)
                correct += equals.sum().item()

                if dataset_types[index] == "probability":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, max_index.unsqueeze(1), 1)
                    scores[scores > 0] = 1
                    correct_predict += (scores * mask).sum().item()
                elif dataset_types[index] == "multi_attempt":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, max_index.unsqueeze(1), 1)
                    correct_predict += (scores * mask).sum().item()

            acc_predict = correct_predict/len(test_dataset)
            acc = correct/len(test_dataset)
            print(f"acc_{data_path}:", acc_predict)
            print("acc", acc)
            result[data_path] = [acc, acc_predict]
    return result


if __name__ == '__main__': 
    device = "cuda:2"
    parser = argparse.ArgumentParser(description="the training code for router")
    writer = SummaryWriter(log_dir='./logs/experiment/experiment_window/experiment_select_base_0.92_2000')
    # dataset and path
    parser.add_argument('--data_paths', nargs='+', default=["./datasets/split2_model7_cluster/gsm8k-train.json","./datasets/split2_model7_cluster/humaneval_train.json", "./datasets/split2_model7_cluster/arc_challenge_train.json", "./datasets/split2_model7_cluster/mmlu_train.json","./datasets/split2_model7_cluster/cmmlu_train.json",])
    parser.add_argument('--test_data_paths',nargs='+', default=["./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json", "./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--test_data_type', nargs='+', default=["multi_attempt", "multi_attempt", "probability", "probability", "probability"])
    parser.add_argument('--final_eval_data_paths', default=["./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/MATH_prealgebra.json", "./datasets/split2_model7/mbpp.json", "./datasets/split2_model7/ceval.json" ,"./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json",  "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--final_eval_data_type', nargs='+', default=["probability", "probability", "multi_attempt","probability", "multi_attempt", "multi_attempt", "probability",  "probability"])

    # training paras
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=2000)
    parser.add_argument('--eval_steps',type=int,default=50)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--save_path', type=str, default='logs/models/model_select_base_0.92_2000')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--last_k',type=int, default=3)
    parser.add_argument('--tempreture', type=int, default=0.2)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--sample_loss_weight', type=float, default=0.01)
    parser.add_argument('--cluster_loss_weight', type=float, default=0.01)
    parser.add_argument('--H', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training_samples_per_dataset', type=int, default=10000)
    parser.add_argument('--window_size', type=int, default=4, help='滑动窗口大小')  # 新增参数
    # final_eval
    parser.add_argument('--final_eval', action="store_true")
    parser.add_argument('--similarity_threshold',type=float,default=0.95)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    print(args.similarity_threshold)
    # get router model (mdeberta-v3-base)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side='left', padding=True)
    encoder_model = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")
   
    # get the training data (x, y)
        # get the training data (x, y)
    router_datasets = [RouterDataset(
        data_path,
        size=args.training_samples_per_dataset,
        data_type=args.test_data_type[i],
        dataset_id=i,
        window_size=args.window_size ,
        similarity_threshold=args.similarity_threshold 
    ) for i, data_path in enumerate(args.data_paths)]
    for router_dataset in router_datasets:
        router_dataset.register_tokenizer(tokenizer)

    router_dataset = ConcatDataset(router_datasets)

    print(f"init_model, router_node size: {router_datasets[0].router_node}")
    router_model = RouterModule(encoder_model, hidden_state_dim=768, node_size=len(router_datasets[0].router_node), similarity_function=args.similarity_function).to(device)

    # get the optimizer (AdamW)
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=args.learning_rate)

    # start training
    print("Training start!!!")
    pbar = tqdm(range(args.training_steps))
    step = 0
    training_log = []
    max_average = 0
    max_training_average = 0

    while(True):
        losses = AverageMeter('Loss', ':3.2f')
        data_loader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, scores, dataset_ids, cluster_ids = batch
            inputs = inputs.to(device)
            scores = scores.to(device)
            dataset_ids = dataset_ids.to(device)
            cluster_ids = cluster_ids.to(device)
            # normalize the target scores

            x, hidden_state = router_model.forward(t=args.tempreture, **inputs)
            loss = router_model.compute_sample_llm_loss(x = x, index_true=scores, top_k = args.top_k, last_k = args.last_k)
            # 记录损失值
         
            if args.sample_loss_weight:
                sample_sample_loss = router_model.compute_sample_sample_loss_with_task_tag(hidden_state=hidden_state, dataset_ids=dataset_ids, t=args.tempreture, H=args.H)
                loss = loss + args.sample_loss_weight * sample_sample_loss

            if args.cluster_loss_weight:
                cluster_loss = router_model.compute_cluster_loss(hidden_state=hidden_state, cluster_ids=cluster_ids, t=args.tempreture, H=args.H)
                loss = loss + args.cluster_loss_weight * cluster_loss

            losses.update(loss.item(), scores.size(0))
            loss.backward()
            if step % args.gradient_accumulation == 0:   
                optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), step)
            pbar.set_postfix({"step": f"{step}","loss": loss.item()})
            pbar.write(f"step:{step}, loss:{loss.item()}")
            pbar.update(1)
            step += 1
            if step >= args.training_steps:
                break
            if (step + 1) % args.eval_steps == 0:
                print("validation start")
                val_result = evaluation(router_model, args.data_paths, args.test_data_type, tokenizer, batch_size = args.batch_size, device=device,similarity=args.similarity_threshold)
                print("test start")
                test_result = evaluation(router_model, args.test_data_paths, args.test_data_type, tokenizer, batch_size = args.batch_size, device=device,similarity=args.similarity_threshold)
                for metric_name, metric_value in val_result.items():
                    writer.add_scalar(f'Validation/{metric_name}', metric_value[1], step)

                for metric_name, metric_value in test_result.items():
                    writer.add_scalar(f'Test/{metric_name}', metric_value[1], step)

                result = {**val_result, **test_result}
                average = sum([ value[1] for value in test_result.values()]) / len(test_result)
                print("average testing", average)
                if average > max_average:
                    torch.save(router_model.state_dict(),  os.path.join(args.save_path, "best_model.pth"))
                    max_average = average
                training_log.append(result)
                training_average = sum([ value[1] for value in val_result.values()]) / len(test_result)
                print("average training", training_average)
                if training_average > max_training_average:
                    torch.save(router_model.state_dict(),  os.path.join(args.save_path, "best_training_model.pth"))
                    max_training_average = training_average

        print(f"step:{step}, avg_loss_per_epoch:{losses.avg}")
        if step >= args.training_steps:
            break

    if args.final_eval:
        state_dict = torch.load(os.path.join(args.save_path, "best_training_model_window.pth"))
        router_model.load_state_dict(state_dict)
        print("test start")
        test_result = evaluation(router_model, args.final_eval_data_paths, args.final_eval_data_type, tokenizer, batch_size=32, device="cuda")
        print(test_result)
           
        output_order = ['mmlu', 'gsm8k', 'cmmlu', 'arc', 'humaneval', 'MATH', 'mbpp', 'ceval']
        key_list = list(test_result.keys())
        key_order = []
        for key_candidate in output_order:
            for key in key_list:
                if key_candidate in key:
                    key_order.append(key)
                    break
        for key in key_order:
            print(f"{test_result[key][1] * 100}", end=' ')

    print("best avg", max_average)
    print("best training avg", max_training_average)

    # save the model
    with open(os.path.join(args.save_path, "training_log_window.json"), 'w') as f:
        json.dump(training_log, f)

    with open(os.path.join(args.save_path, "config.txt"), 'w') as f:
        f.write(str(args))
    