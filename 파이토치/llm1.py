import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 모델과 토크나이저 불러오기
# KoGPT 모델의 이름
model_name = "skt/kogpt2-base-v2"

# 토크나이저는 텍스트를 모델이 이해할 수 있는 숫자 시퀀스로 변환하는 역할을 합니다.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# AutoModelForCausalLM은 텍스트 생성에 최적화된 모델 클래스입니다.
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 텍스트 생성 설정
# 모델을 평가(추론) 모드로 전환
model.eval()

# GPU가 있다면 GPU로 모델을 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 입력 텍스트 준비
# 모델에 입력할 텍스트입니다.
input_text = "안녕하세요. 저는 오늘"

# 토크나이저를 사용해 입력 텍스트를 모델 입력 형식(토큰 ID)으로 변환
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 4. 텍스트 생성
# 모델의 `generate` 메서드를 사용해 텍스트를 생성합니다.
# max_length: 생성할 문장의 최대 길이 (입력 텍스트 포함)
# do_sample: True로 설정하면 다양한 문장을 생성합니다.
# top_k: 가장 확률 높은 k개의 토큰 중에서 샘플링합니다.
# temperature: 온도를 낮추면 더 안정적인 문장을, 높이면 더 창의적인 문장을 생성합니다.
# no_repeat_ngram_size: 반복되는 n-그램을 방지합니다.
generated_ids = model.generate(
    input_ids,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id, # 문장의 끝을 알리는 토큰
    repetition_penalty=2.0,
    do_sample=True,
    top_k=50,
    temperature=0.9
)

# 5. 생성된 텍스트 디코딩 및 출력
# 생성된 토큰 ID를 다시 사람이 읽을 수 있는 텍스트로 변환
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("--- 입력 텍스트 ---")
print(input_text)
print("\n--- 생성된 텍스트 ---")
print(generated_text)