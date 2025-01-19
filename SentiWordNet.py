import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 下载NLTK的必要资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# 加载预训练的句子嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级且高效的模型

def load_sentiwordnet(file_path):
    sentiwordnet = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            # 跳过注释行
            if line.startswith("#"):
                continue

            # 按制表符分割行，提取相关字段
            parts = line.strip().split("\t")
            if len(parts) < 6:
                print(f"Skipping line {line_num}: 不符合规范的行")
                continue  # 跳过格式不正确的行

            pos = parts[0].strip()  # 词性
            synset_id = parts[1].strip()  # ID
            try:
                pos_score = float(parts[2].strip())  # 正向情感得分
            except ValueError:
                pos_score = 0.0
                print(f"Warning: Line {line_num} 正向得分转换失败，设为0.0")
            try:
                neg_score = float(parts[3].strip())  # 负向情感得分
            except ValueError:
                neg_score = 0.0
                print(f"Warning: Line {line_num} 负向得分转换失败，设为0.0")
            gloss = parts[5].strip()  # 定义
            words_field = parts[4].strip()  # SynsetTerms

            # 计算客观情感得分
            obj_score = 1.0 - (pos_score + neg_score)

            # 提取同义词集词汇
            words = words_field.split()

            for word in words:
                # 检查是否包含'#'
                if "#" not in word:
                    print(f"Warning: Line {line_num} 的词 '{word}' 不包含 '#'，跳过此词")
                    continue

                base_word, synset_num = word.split("#", 1)  # 提取基本词汇和编号

                if not synset_num.isdigit():
                    print(f"Warning: Line {line_num} 的词 '{word}' 编号部分不是数字，跳过此词")
                    continue

                if base_word not in sentiwordnet:
                    sentiwordnet[base_word] = {}

                # 将同义词集信息存储到字典中
                sentiwordnet[base_word][synset_num] = {
                    "pos_score": pos_score,
                    "neg_score": neg_score,
                    "obj_score": obj_score,
                    "gloss": gloss,
                    "pos": pos  # 词性
                }

    return sentiwordnet


# def extract_emotion_words(text):
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text.lower(), language='english', preserve_line=True)
#     filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
#     return filtered_words
def extract_adjectives(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower(), language='english', preserve_line=True)
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # 标记每个单词的词性
    tagged_words = pos_tag(filtered_words)

    # 只保留形容词（JJ、JJR、JJS为形容词的标记）
    adjectives = [word for word, tag in tagged_words if tag in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']]

    return adjectives


def get_sentiment_score(word, sentiwordnet, model, context_embedding):
    if word in sentiwordnet:
        synsets = sentiwordnet[word]
        glosses = [synset["gloss"] for synset in synsets.values()]

        # 生成 gloss 的嵌入
        gloss_embeddings = model.encode(glosses, show_progress_bar=False)

        # 计算上下文与每个 gloss 的余弦相似度
        similarities = cosine_similarity([context_embedding], gloss_embeddings)[0]

        # 选择相似度最高的同义词集
        best_synset_idx = np.argmax(similarities)
        best_synset_num = list(synsets.keys())[best_synset_idx]
        best_synset = synsets[best_synset_num]

        return best_synset["pos_score"], best_synset["neg_score"], best_synset["obj_score"]
    else:
        return 0, 0, 1  # 如果词汇不在SentiWordNet中，则返回中性得分

sentiwordnet = load_sentiwordnet('SentiWordNet_3.0.0_20130122.txt')
def main():
    # 加载SentiWordNet词典
    sentiwordnet = load_sentiwordnet('SentiWordNet_3.0.0_20130122.txt')
    # 读取txt文件并提取评论部分
    file_path = './data/rest16/train.txt'  # 请替换为你的文件路径

    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行

    # 提取每行的评论部分
    reviews = [line.split('####')[0].strip() for line in lines]  # 分割并去除多余的空格

    # 打印前5个评论（检查结果）
    for review in reviews[:10]:
        print(review)

    # 处理数据集中的每条评论
    for index, review in enumerate(reviews[:20]):
        emotion_words = extract_adjectives(review)
        print(f"\nProcessing Review {index}: {review}")

        # 生成句子的上下文嵌入
        context_embedding = model.encode(review)

        # 获取情感分数并使用上下文
        for word in emotion_words:
            pos_score, neg_score, obj_score = get_sentiment_score(word, sentiwordnet, model, context_embedding)
            print(
                f"Word: {word}, Positive Score: {pos_score}, Negative Score: {neg_score}, Objective Score: {obj_score}")

    # # 打印转换后的数据结构（示例）
    # for word, synsets in list(sentiwordnet.items())[:5]:  # 打印前5个词汇的转换结果
    #     print(f"{word}:")
    #     for num, details in synsets.items():
    #         print(f"  {num}: {details}")
    #
    # # 示例：从评论中提取情感相关词汇
    # review = "The food was delicious but the service was terrible."
    # emotion_words = extract_adjectives(review)
    # print("\nEmotion Words:", emotion_words)
    #
    # # 加载预训练的句子嵌入模型
    # model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级且高效的模型
    #
    # # 生成上下文的嵌入（整个句子的嵌入）
    # context_embedding = model.encode(review)
    #
    # # 获取情感分数并使用上下文
    # for word in emotion_words:
    #     pos_score, neg_score, obj_score = get_sentiment_score(word, sentiwordnet, model, context_embedding)
    #     print(f"Word: {word}, Positive Score: {pos_score}, Negative Score: {neg_score}, Objective Score: {obj_score}")


if __name__ == "__main__":
    main()

# from symtable import Class
#
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
#
# # 下载NLTK的必要资源
# nltk.download('punkt')
# nltk.download('stopwords')
# def load_sentiwordnet(file_path):
#     sentiwordnet = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             # 跳过注释行
#             if line.startswith("#"):
#                 continue
#             parts = line.split("\t")
#             if len(parts) < 5:
#                 print("格式不符合规范的行",line)
#                 continue  # 如果数据格式不完整则跳过此行
#             # 提取正面情感分数、负面情感分数
#             pos_score = parts[2].strip()  # 正面情感分数
#             neg_score = parts[3].strip()  # 负面情感分数
#             words = parts[4].strip()  # 词汇（含有词形ID）
#             # 提取词汇，如“able#1”，我们只关心“able”
#             word_list = words.split(" ")
#
#             # 尝试将情感分数转换为浮动数值
#             # 尝试将情感分数转换为浮动数值
#             try:
#                 pos_score = float(pos_score)
#             except ValueError:
#                 pos_score = 0.0  # 如果转换失败，设为默认值
#                 print(f"Warning: Could not convert '{pos_score}' to float. Returning default value {0}.")
#                 print("Could not convert line",line)
#             try:
#                 neg_score = float(neg_score)
#             except ValueError:
#                 neg_score = 0.0  # 如果转换失败，设为默认值
#                 print(f"Warning: Could not convert '{pos_score}' to float. Returning default value {0}.")
#                 print("Could not convert line", line)
#
#             # 计算客观情感分数
#             obj_score = 1.0 - (pos_score + neg_score)
#             for word in word_list:
#                 # 提取词汇的基本部分（如"able"），忽略词形编号（如#1）
#                 base_word = word.split("#")[0]
#                 sentiwordnet[base_word] = {
#                     "pos_score": pos_score,
#                     "neg_score": neg_score,
#                     "obj_score": obj_score
#                 }
#     return sentiwordnet
# def extract_emotion_words(text):
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text.lower(),language='english', preserve_line=True)
#     filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
#     return filtered_words
#
# # 从SentiWordNet获取情感分数
# def get_sentiment_score(word, sentiwordnet):
#     if word in sentiwordnet:
#         sentiment = sentiwordnet[word]
#         return sentiment["pos_score"], sentiment["neg_score"], sentiment["obj_score"]
#     else:
#         return 0, 0, 1  # 如果词汇不在SentiWordNet中，则返回中性得分
#
# # 示例：加载SentiWordNet词典
# sentiwordnet = load_sentiwordnet('SentiWordNet_3.0.0_20130122.txt')
# # 打印字典的前5条记录
# for i, (word, scores) in enumerate(sentiwordnet.items()):
#     if i < 5:
#         print(f"{word}: {scores}")
#     else:
#         break
# # 示例：从评论中提取情感相关词汇
# review = "The food was delicious but the service was terrible."
# emotion_words = extract_emotion_words(review)
# print("Emotion Words:", emotion_words)
# # 示例：获取情感分数
# for word in emotion_words:
#     pos_score, neg_score, obj_score = get_sentiment_score(word, sentiwordnet)
#     print(f"Word: {word}, Positive Score: {pos_score}, Negative Score: {neg_score}, Objective Score: {obj_score}")
#
