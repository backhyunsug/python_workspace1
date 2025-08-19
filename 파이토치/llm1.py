import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. �𵨰� ��ũ������ �ҷ�����
# KoGPT ���� �̸�
model_name = "skt/kogpt2-base-v2"

# ��ũ�������� �ؽ�Ʈ�� ���� ������ �� �ִ� ���� �������� ��ȯ�ϴ� ������ �մϴ�.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# AutoModelForCausalLM�� �ؽ�Ʈ ������ ����ȭ�� �� Ŭ�����Դϴ�.
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. �ؽ�Ʈ ���� ����
# ���� ��(�߷�) ���� ��ȯ
model.eval()

# GPU�� �ִٸ� GPU�� ���� �̵�
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. �Է� �ؽ�Ʈ �غ�
# �𵨿� �Է��� �ؽ�Ʈ�Դϴ�.
input_text = "�ȳ��ϼ���. ���� ����"

# ��ũ�������� ����� �Է� �ؽ�Ʈ�� �� �Է� ����(��ū ID)���� ��ȯ
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 4. �ؽ�Ʈ ����
# ���� `generate` �޼��带 ����� �ؽ�Ʈ�� �����մϴ�.
# max_length: ������ ������ �ִ� ���� (�Է� �ؽ�Ʈ ����)
# do_sample: True�� �����ϸ� �پ��� ������ �����մϴ�.
# top_k: ���� Ȯ�� ���� k���� ��ū �߿��� ���ø��մϴ�.
# temperature: �µ��� ���߸� �� �������� ������, ���̸� �� â������ ������ �����մϴ�.
# no_repeat_ngram_size: �ݺ��Ǵ� n-�׷��� �����մϴ�.
generated_ids = model.generate(
    input_ids,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id, # ������ ���� �˸��� ��ū
    repetition_penalty=2.0,
    do_sample=True,
    top_k=50,
    temperature=0.9
)

# 5. ������ �ؽ�Ʈ ���ڵ� �� ���
# ������ ��ū ID�� �ٽ� ����� ���� �� �ִ� �ؽ�Ʈ�� ��ȯ
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("--- �Է� �ؽ�Ʈ ---")
print(input_text)
print("\n--- ������ �ؽ�Ʈ ---")
print(generated_text)