import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from stanza.server import CoreNLPClient
from stanza.server import StartServer

class DependencyTreeCollate:
    def __init__(self, stanford_url="http://localhost:9000", max_len=512, padding_value=0):
        self.stanford_url = stanford_url
        self.max_len = max_len
        self.padding_value = padding_value

    def __call__(self, batch):
        source_ids = [item["source_ids"].clone().detach() for item in batch]
        source_masks = [item["source_mask"].clone().detach() for item in batch]
        target_ids = [item["target_ids"].clone().detach() for item in batch]
        target_masks = [item["target_mask"].clone().detach() for item in batch]

        if "dependency_tree" in batch[0]:
            dependency_trees = [item["dependency_tree"].clone().detach() for item in batch]
        else:
            dependency_trees = self._generate_dependency_trees(batch)

        source_ids_padded = pad_sequence(source_ids, batch_first=True, padding_value=self.padding_value)
        source_masks_padded = pad_sequence(source_masks, batch_first=True, padding_value=self.padding_value)
        target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=self.padding_value)
        target_masks_padded = pad_sequence(target_masks, batch_first=True, padding_value=self.padding_value)
        dependency_trees_padded = self._pad_dependency_trees(dependency_trees)

        return {
            "source_ids": source_ids_padded,
            "source_mask": source_masks_padded,
            "target_ids": target_ids_padded,
            "target_mask": target_masks_padded,
            "dependency_tree": dependency_trees_padded
        }

    def _generate_dependency_trees(self, batch):
        with CoreNLPClient(
            endpoint=self.stanford_url,
            start_server=StartServer.DONT_START,
            annotators=["tokenize", "ssplit", "pos", "parse", "depparse"],
            timeout=60000,
            memory="4G"
        ) as client:
            sentences = [item["sentence"] for item in batch]
            dependency_trees = []

            for sentence in sentences:
                try:
                    ann = client.annotate(sentence)
                    if len(ann.sentence) > 0 and hasattr(ann.sentence[0], 'token'):
                        dep_tree = self._convert_dependency_to_matrix(ann.sentence[0].token)
                        dependency_trees.append(dep_tree)
                    else:
                        print(f"Warning: No dependency data for sentence: {sentence}")
                        dependency_trees.append(torch.zeros((self.max_len, self.max_len), dtype=torch.float32))
                except Exception as e:
                    print(f"Error processing sentence: {sentence}, {e}")
                    dependency_trees.append(torch.zeros((self.max_len, self.max_len), dtype=torch.float32))

            return dependency_trees

    def _convert_dependency_to_matrix(self, tokens):
        matrix = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        for token in tokens:
            if hasattr(token, 'dependencyEdge') and hasattr(token.dependencyEdge, 'target') and hasattr(token.dependencyEdge, 'source'):
                target, source = token.dependencyEdge.target, token.dependencyEdge.source
                if target < self.max_len and source < self.max_len:
                    matrix[target, source] = 1.0
        return torch.tensor(matrix, dtype=torch.float32)

    def _pad_dependency_trees(self, dependency_trees):
        if not dependency_trees:
            raise ValueError("dependency_trees is empty. Cannot pad empty dependency trees.")

        max_tree_len = max(tree.size(0) for tree in dependency_trees)
        padded_trees = torch.stack([
            torch.nn.functional.pad(tree, (0, max_tree_len - tree.size(0), 0, max_tree_len - tree.size(1)))
            for tree in dependency_trees
        ])
        return padded_trees

# import torch
# from torch.nn.utils.rnn import pad_sequence
# import numpy as np
# from stanza.server import CoreNLPClient
# from stanza.server import StartServer

# class DependencyTreeCollate:
#     def __init__(self, stanford_url="http://localhost:9000", max_len=512, padding_value=0):
#         self.stanford_url = stanford_url
#         self.max_len = max_len
#         self.padding_value = padding_value

#     def __call__(self, batch):
#         """
#         处理输入批次，返回填充后的张量。
#         """
#         # 分别处理 source 和 target 数据
#         source_ids = [torch.tensor(item["source_ids"], dtype=torch.long) for item in batch]
#         source_masks = [torch.tensor(item["source_mask"], dtype=torch.long) for item in batch]
#         target_ids = [torch.tensor(item["target_ids"], dtype=torch.long) for item in batch]
#         target_masks = [torch.tensor(item["target_mask"], dtype=torch.long) for item in batch]

#         # 动态解析依存树或直接处理已有的依存树
#         if "dependency_tree" in batch[0]:
#             dependency_trees = [torch.tensor(item["dependency_tree"], dtype=torch.float32) for item in batch]
#         else:
#             dependency_trees = self._generate_dependency_trees(batch)

#         # 填充所有字段
#         source_ids_padded = pad_sequence(source_ids, batch_first=True, padding_value=self.padding_value)
#         source_masks_padded = pad_sequence(source_masks, batch_first=True, padding_value=self.padding_value)
#         target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=self.padding_value)
#         target_masks_padded = pad_sequence(target_masks, batch_first=True, padding_value=self.padding_value)
#         dependency_trees_padded = self._pad_dependency_trees(dependency_trees)

#         return {
#             "source_ids": source_ids_padded,
#             "source_mask": source_masks_padded,
#             "target_ids": target_ids_padded,
#             "target_mask": target_masks_padded,
#             "dependency_tree": dependency_trees_padded
#         }
#     def _generate_dependency_trees(self, batch):
#     # """
#     # 使用 CoreNLP 服务动态生成依存树。
#     # """
#         with CoreNLPClient(
#             endpoint=self.stanford_url,
#             start_server=StartServer.DONT_START,          # 不尝试启动新服务
#             annotators=["tokenize", "ssplit", "pos", "parse", "depparse"],
#             timeout=60000,
#             memory="4G"
#         ) as client:
#             sentences = [item["sentence"] for item in batch]
#             print("oooooo"*30)
#             print(sentences)
#             dependency_trees = []
#             for sentence in sentences:
#                 try:
#                     ann = client.annotate(sentence)
#                     # print(f"Processed sentence: {sentence}")
#                     # print("-ann"*30)
#                     # print(f"Annotation result: {ann}")
                
#                     # 检查是否成功生成依存关系
#                     if len(ann.sentence) > 0 and hasattr(ann.sentence[0], 'token'):
#                         dep_tree = self._convert_dependency_to_matrix(ann.sentence[0].token)
#                         dependency_trees.append(dep_tree)
#                     else:
#                         print("No dependencyEdge found in the sentence.")
#                 except Exception as e:
#                     print(f"Error processing sentence: {sentence}")
#                     print(e)
#             return dependency_trees

#     # def _generate_dependency_trees(self, batch):
#     #     """
#     #     使用 CoreNLP 服务动态生成依存树。
#     #     """
#     #     with CoreNLPClient(
#     #         endpoint=self.stanford_url,
#     #         start_server=StartServer.DONT_START,          # 不尝试启动新服务
#     #         annotators=["tokenize", "ssplit", "pos", "parse", "depparse"],
#     #         timeout=60000,
#     #         memory="4G"
#     #     ) as client:
#     #         sentences = [item["sentence"] for item in batch]
#     #         dependency_trees = []
#     #         for sentence in sentences:
#     #             ann = client.annotate(sentence)
#     #             dep_tree = self._convert_dependency_to_matrix(ann.sentence[0].token)
#     #             dependency_trees.append(dep_tree)
#     #         return dependency_trees

#     def _convert_dependency_to_matrix(self, tokens):
#         """
#         将解析结果转换为邻接矩阵表示。
#         """
#         matrix = np.zeros((self.max_len, self.max_len), dtype=np.float32)
#         for token in tokens:
#             print("*"*30)
#             print(token)
#             print("*"*20)
#             print(hasattr(token, 'dependencyEdge'))

#             if token.dependencyEdge.target < self.max_len and token.dependencyEdge.source < self.max_len:
#                 matrix[token.dependencyEdge.target, token.dependencyEdge.source] = 1.0
#         return torch.tensor(matrix, dtype=torch.float32)

#     def _pad_dependency_trees(self, dependency_trees):
#         """
#         对依存树邻接矩阵进行填充。
#         """
#         max_tree_len = max(tree.size(0) for tree in dependency_trees)
#         padded_trees = torch.stack([
#             torch.nn.functional.pad(tree, (0, max_tree_len - tree.size(0), 0, max_tree_len - tree.size(1)))
#             for tree in dependency_trees
#         ])
#         return padded_trees
