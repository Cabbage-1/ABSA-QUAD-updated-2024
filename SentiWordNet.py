from symtable import Class

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 下载NLTK的必要资源
nltk.download('punkt')
nltk.download('stopwords')
def load_sentiwordnet(file_path):
    sentiwordnet = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 跳过注释行
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                print("格式不符合规范的行",line)
                continue  # 如果数据格式不完整则跳过此行
            # 提取正面情感分数、负面情感分数
            pos_score = parts[2].strip()  # 正面情感分数
            neg_score = parts[3].strip()  # 负面情感分数
            words = parts[4].strip()  # 词汇（含有词形ID）
            # 提取词汇，如“able#1”，我们只关心“able”
            word_list = words.split(" ")

            # 尝试将情感分数转换为浮动数值
            # 尝试将情感分数转换为浮动数值
            try:
                pos_score = float(pos_score)
            except ValueError:
                pos_score = 0.0  # 如果转换失败，设为默认值
                print(f"Warning: Could not convert '{pos_score}' to float. Returning default value {0}.")
                print("Could not convert line",line)
            try:
                neg_score = float(neg_score)
            except ValueError:
                neg_score = 0.0  # 如果转换失败，设为默认值
                print(f"Warning: Could not convert '{pos_score}' to float. Returning default value {0}.")
                print("Could not convert line", line)

            # 计算客观情感分数
            obj_score = 1.0 - (pos_score + neg_score)
            for word in word_list:
                # 提取词汇的基本部分（如"able"），忽略词形编号（如#1）
                base_word = word.split("#")[0]
                sentiwordnet[base_word] = {
                    "pos_score": pos_score,
                    "neg_score": neg_score,
                    "obj_score": obj_score
                }
    return sentiwordnet
def extract_emotion_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower(),language='english', preserve_line=True)
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words

# 从SentiWordNet获取情感分数
def get_sentiment_score(word, sentiwordnet):
    if word in sentiwordnet:
        sentiment = sentiwordnet[word]
        return sentiment["pos_score"], sentiment["neg_score"], sentiment["obj_score"]
    else:
        return 0, 0, 1  # 如果词汇不在SentiWordNet中，则返回中性得分

# 示例：加载SentiWordNet词典
sentiwordnet = load_sentiwordnet('SentiWordNet_3.0.0_20130122.txt')
# 打印字典的前5条记录
for i, (word, scores) in enumerate(sentiwordnet.items()):
    if i < 5:
        print(f"{word}: {scores}")
    else:
        break
# 示例：从评论中提取情感相关词汇
review = "The food was delicious but the service was terrible."
emotion_words = extract_emotion_words(review)
print("Emotion Words:", emotion_words)
# 示例：获取情感分数
for word in emotion_words:
    pos_score, neg_score, obj_score = get_sentiment_score(word, sentiwordnet)
    print(f"Word: {word}, Positive Score: {pos_score}, Negative Score: {neg_score}, Objective Score: {obj_score}")

