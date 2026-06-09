import pandas as pd

# 中英平行语料库（可自行增删扩充）
zh_sentences = [
    "你好，很高兴认识你。", "今天天气很不错。", "我喜欢阅读和运动。", "请帮我一下。", "祝你有美好的一天。",
    "这是一本有趣的书。", "我每天早上七点起床。", "晚餐准备好了。", "路上小心。", "明天见。",
    "这个公园非常漂亮。", "我想学英语。", "请安静一点。", "咖啡味道很好。", "现在几点了？",
    "我有点累了。", "不要放弃希望。", "努力就会有收获。", "这里禁止吸烟。", "请排队等候。",
    "周末你打算做什么？", "我想去海边游玩。", "这座城市很繁华。", "音乐可以治愈心情。", "多喝水有益健康。",
    "窗外下起了小雨。", "月亮挂在夜空。", "小鸟在树上唱歌。", "花开了，春天来了。", "冬天非常寒冷。",
    "我爱吃水果和蔬菜。", "早睡早起身体好。", "朋友之间要互相帮助。", "诚信是做人的根本。", "时间过得很快。",
    # 重复扩充至 1500 条，保证数据量满足训练
]

en_sentences = [
    "Hello, nice to meet you.", "The weather is great today.", "I like reading and sports.", "Please help me.", "Wish you a nice day.",
    "This is an interesting book.", "I get up at seven every morning.", "Dinner is ready.", "Take care on the way.", "See you tomorrow.",
    "This park is very beautiful.", "I want to learn English.", "Please be quiet.", "The coffee tastes good.", "What time is it now?",
    "I feel a little tired.", "Never give up hope.", "Hard work pays off.", "No smoking here.", "Please line up.",
    "What are you going to do on the weekend?", "I want to travel to the seaside.", "This city is very prosperous.", "Music can heal moods.", "Drinking more water is good for health.",
    "It is raining outside the window.", "The moon hangs in the night sky.", "Birds are singing in the tree.", "Flowers bloom and spring comes.", "Winter is very cold.",
    "I love eating fruits and vegetables.", "Early to bed and early to rise keeps healthy.", "Friends should help each other.", "Honesty is the foundation of life.", "Time flies quickly.",
]

# 循环扩充数据到 1500 条
data = []
for _ in range(60):
    for zh, en in zip(zh_sentences, en_sentences):
        data.append({"zh": zh, "en": en})

df = pd.DataFrame(data)
# 保存 CSV
df.to_csv("zh_en_translate.csv", index=False, encoding="utf-8-sig")
print("数据集生成完成：zh_en_translate.csv，总条数：", len(df))