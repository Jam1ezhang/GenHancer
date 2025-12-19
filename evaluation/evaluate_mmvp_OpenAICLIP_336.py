import os
import csv
from PIL import Image
import torch
from tqdm import tqdm
import json
from transformers import CLIPVisionModel, CLIPModel, CLIPImageProcessor, CLIPTokenizer



def benchmark_model(processor, tokenizer, model, benchmark_dir, device="cpu"):

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    csv_outfile = open('Prediction_Results_OpenAICLIP', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            text1 = tokenizer(
                text1,
                truncation=True,
                max_length=77,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].to(device)
            text2 = tokenizer(
                text2,
                truncation=True,
                max_length=77,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].to(device)   # torch.Size([1, 77])

            img1 = processor.preprocess(img1, return_tensors='pt')['pixel_values'].to(device)
            img2 = processor.preprocess(img2, return_tensors='pt')['pixel_values'].to(device)
            imgs = torch.cat((img1, img2), dim=0)

            with torch.no_grad():
                model.eval().float()

                outputs1 = model(input_ids=text1, pixel_values=imgs)
                logits_per_image1, logits_per_text1 = outputs1.logits_per_image, outputs1.logits_per_text
                outputs2 = model(input_ids=text2, pixel_values=imgs)
                logits_per_image2, logits_per_text2 = outputs2.logits_per_image, outputs2.logits_per_text
                
                probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1
            
        csv_outfile.close()

    # Calculate percentage accuracies
    Category_Score_List = []
    
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100
        Category_Score_List.append(pair_accuracies[category])
        
    pair_accuracies['average_score'] = sum(Category_Score_List)/len(Category_Score_List)

    return pair_accuracies


def official_evaluation(processor, tokenizer, clip_model, model_name, benchmark_dir, device):
    
    with torch.no_grad():
        clip_model.eval()

        results_openai = {f'{model_name}': benchmark_model(processor, tokenizer, clip_model, benchmark_dir, device)}

        # Merge results
        results = {**results_openai}

        # Convert results to format suitable for star plot
        categories = results[list(results.keys())[0]].keys()
        data = {'Categories': list(categories)}
        for model in list(results_openai.keys()):
            data[model] = [results[model][category] for category in categories]

        return results


if __name__ == "__main__":

    BENCHMARK_DIR = '/home/user/gptdata/zym/codespace_hallucination/video_data/MMVP_VLM'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------- 模型加载配置 --------------------- 
    # 原始CLIP模型名称 - 用于加载image_processor和tokenizer
    original_clip_model = '/home/user/gptdata/zym/codespace_hallucination/ckpts/clip-vit-large-patch14-336'
    
    # 指定的权重文件夹 - 只用于加载模型权重
    custom_weights_dir = f'/home/user/gptdata/zym/codespace_hallucination/ckpts/clip-vit-large-patch14-336'
    # /home/user/gptdata/zym/codespace_hallucination/GenHancer/Continuous/output_OpenAICLIP_336_video_stage2_all_load626/clip-vit-large-patch14-336-800
    # 友好的模型名称 - 用于结果展示
    friendly_model_name = 'OpenAICLIP_336_Video_Stage2'
    # ------------------------------------------------------
    
    # Step 1: 加载原始CLIP的image_processor和tokenizer
    print(f"正在从{original_clip_model}加载image_processor和tokenizer...")
    image_processor = CLIPImageProcessor.from_pretrained(original_clip_model)
    tokenizer = CLIPTokenizer.from_pretrained(original_clip_model, max_length=77)
    print("✅ image_processor和tokenizer加载完成")
    
    # Step 2: 加载模型结构并尝试加载自定义权重
    print(f"正在从{original_clip_model}加载模型结构...")
    vision_tower = CLIPModel.from_pretrained(original_clip_model, device_map=device)
    
    try:
        print(f"尝试从{custom_weights_dir}加载自定义权重...")
        # 尝试使用HuggingFace的from_pretrained方法加载完整模型
        vision_tower = CLIPModel.from_pretrained(custom_weights_dir, device_map=device)
        print("✅ 成功加载自定义完整模型")
    except Exception as e:
        print(f"❌ 无法加载完整模型：{e}")
        print("尝试直接加载权重文件...")
        
        # 检查权重文件是否存在
        weights_file = os.path.join(custom_weights_dir, 'pytorch_model.bin')
        if os.path.exists(weights_file):
            try:
                # 直接加载权重文件
                state_dict = torch.load(weights_file, map_location=device)
                
                # 处理可能的键名不匹配问题
                if any(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {k[6:]: v for k, v in state_dict.items()}
                
                # 加载权重，strict=False允许部分权重不匹配
                vision_tower.load_state_dict(state_dict, strict=False)
                print(f"✅ 成功从{weights_file}加载自定义权重")
            except Exception as load_error:
                print(f"❌ 加载权重文件失败：{load_error}")
                print("⚠️  将使用原始CLIP模型权重")
        else:
            print(f"❌ 权重文件不存在：{weights_file}")
            print("⚠️  将使用原始CLIP模型权重")
    
    print(f"模型设备：{device}")
    print("模型加载完成，开始评估...")
    
    # Step 3: 执行评估
    results = official_evaluation(image_processor, tokenizer, vision_tower, friendly_model_name, BENCHMARK_DIR, device)
    print("\n评估结果：")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for category, accuracy in metrics.items():
            print(f"  {category}: {accuracy:.2f}%")
