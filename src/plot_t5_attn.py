from transformers import T5Tokenizer, T5Model, BartTokenizer, BartModel, PegasusTokenizer, PegasusModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json 
from scipy import stats

def normalize_attention_scores(means, stds):
    total = sum(means.values())
    norm_means = {k: v/total for k, v in means.items()}
    norm_stds = {k: v/total for k, v in stds.items()}
    return norm_means, norm_stds

def analyze_prompt_attention(attention_weights, tokens):
    if not attention_weights or not tokens:
        return None
        
    num_layers = len(attention_weights)
    seq_length = len(tokens)
    
    attention_matrix = np.zeros((num_layers, seq_length))
    for layer_idx in range(num_layers):
        layer_attention = attention_weights[layer_idx][0].detach().cpu().numpy()
        mean_heads = layer_attention.mean(axis=0)
        attention_matrix[layer_idx] = mean_heads.mean(axis=0)
    
    mean_attention = attention_matrix.mean(axis=0)
    
    # Find positions for metadata and x tokens
    x_positions = [i for i, token in enumerate(tokens) if 'x' == token]
    name_positions = [i for i, token in enumerate(tokens) if 'name' in token.lower()]
    desc_positions = [i for i, token in enumerate(tokens) if 'description' in token.lower()]
    obj_positions = [i for i, token in enumerate(tokens) if 'objective' in token.lower()]
    
    num_x = len(x_positions)
    
    if num_x == 0 or not name_positions or not desc_positions or not obj_positions:
        return None
    
    results = {
        
    }
    
    # Calculate metadata section attention scores
    # Name section: tokens between "name" and "description"
    name_start = name_positions[0]
    name_end = desc_positions[0]
    results["name"] = np.mean(mean_attention[name_start:name_end])
    
    # Description section: tokens between "description" and "objective"
    desc_start = desc_positions[0]
    desc_end = obj_positions[0]
    results["description"] = np.mean(mean_attention[desc_start:desc_end])
    
    # Objective section: tokens between "objective" and first x
    obj_start = obj_positions[0]
    obj_end = x_positions[0]
    results["objective"] = np.mean(mean_attention[obj_start:obj_end])

    results["'x' token"] = np.mean([mean_attention[pos] for pos in x_positions])
    
    # Add values between x tokens
    for i in range(num_x - 1):
        start_pos = x_positions[i] + 1
        end_pos = x_positions[i + 1]
        between_tokens = mean_attention[start_pos:end_pos]
        if len(between_tokens) > 0:
            results[f"x{i}"] = np.mean(between_tokens)
    
    after_last_x = mean_attention[x_positions[-1]+1:-1]
    if len(after_last_x) > 0:
        results[f"x{num_x-1}"] = np.mean(after_last_x)
    
    results["End token"] = mean_attention[-1]

    return results

def get_dynamic_xticks(keys, max_ticks=8):
    x_keys = [k for k in keys if k.startswith('x')]
    if len(x_keys) <= max_ticks:
        return keys
    
    fixed_keys = [k for k in keys if not k.startswith('x')]
    step = max(1, len(x_keys) // (max_ticks - len(fixed_keys)))
    selected_x_keys = x_keys[::step]
    
    final_keys = []
    for k in keys:
        if k in fixed_keys:
            final_keys.append(k)
        elif k in selected_x_keys:
            final_keys.append(k)
        else:
            final_keys.append('')
            
    return final_keys

model_name = 't5-small'
device = torch.device('cuda')
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5Model.from_pretrained(model_name).to(device)

name_map = {
    'AntMorphology-Exact-v0': 'Ant',
    'DKittyMorphology-Exact-v0': "D'Kitty",
    'Superconductor-RandomForest-v0': "Superconductor",
    'TFBind8-Exact-v0': "TF Bind 8",
    'TFBind10-Exact-v0': "TF Bind 10",
    'gtopx_data_2_1': "GTOPX 2",
    'gtopx_data_3_1': 'GTOPX 3',
    'gtopx_data_4_1': 'GTOPX 4',
    'gtopx_data_6_1': 'GTOPX 6',
    'RobotPush_100': 'RobotPush',
    'Rover_150': 'Rover',
    'LunarLander_100': 'LunarLander'
}

for task_name in [
    # 'gtopx_data_2_1',
    # 'Superconductor-RandomForest-v0',
    'TFBind10-Exact-v0'
]:
    with open(f"./data/{task_name}.json", 'r') as f:
        data = json.load(f)

    data = data[:100]

    with open(f"./data/{task_name}.metadata", 'r') as f:
        m = f.read()

    texts = [", ".join(d['x']) for d in data]
    texts = [f"{m}. {t}" for t in texts]

    all_results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        decoder_input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)
        
        attention_weights = outputs.encoder_attentions
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # print(tokens)
        
        results = analyze_prompt_attention(attention_weights, tokens)
        # print(results)
        # assert 0
        if results:
            all_results.append(results)

    means = {}
    ci_intervals = {}
    # assert 0, all_results
    for key in all_results[0].keys():
        values = [result[key] for result in all_results if key in result]
        if values:
            means[key] = np.mean(values)
            ci = stats.t.interval(alpha=0.95, 
                                df=len(values)-1,
                                loc=np.mean(values),
                                scale=stats.sem(values))
            ci_intervals[key] = (max(0, ci[0]), ci[1])

    norm_means, norm_ci = normalize_attention_scores(means, ci_intervals)

    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    x = np.arange(len(norm_means))
    means_values = list(norm_means.values())

    yerr_low = [means_values[i] - norm_ci[key][0] for i, key in enumerate(norm_means.keys())]
    yerr_high = [norm_ci[key][1] - means_values[i] for i, key in enumerate(norm_means.keys())]
    yerr = np.array([yerr_low, yerr_high])

    bars = ax.bar(x, means_values, 
                 yerr=yerr,
                 capsize=5, 
                 color='skyblue', 
                 edgecolor='black', 
                 alpha=0.7,
                 error_kw=dict(ecolor='gray', lw=1, capsize=5, capthick=1))

    ax.set_xticks(x)
    xtick_labels = get_dynamic_xticks(list(norm_means.keys()))
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f"Average Attention Distribution of {name_map[task_name]} (#dim = {len(data[0]['x'])})", pad=20)
    ax.set_xlabel('Components')
    ax.set_ylabel('Normalized Attention Score')

    plt.tight_layout()
    
    plt.savefig(f'./attn_plots_tmp/{model_name}_avg_attn_{task_name}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'./attn_plots/{model_name}_avg_attn_{task_name}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
        
    plot_data = {
        "means": means,
        "confidence_intervals": {
            k: [float(ci[0]), float(ci[1])] for k, ci in ci_intervals.items()
        },
        "metadata": {
            "confidence_level": 0.95,
            "sample_size": len(all_results),
            "description": "Average attention scores with 95% confidence intervals"
        }
    }

    with open(f'./attn_data/{model_name}_attn_data_{task_name}.json', 'w') as f:
        json.dump(plot_data, f, indent=4)