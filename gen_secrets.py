import secrets

def generate_secret_key():
    # 生成一个随机的 32 字节（64 字符）的十六进制字符串
    secret_key = secrets.token_hex(32)
    return secret_key

# 生成一个 secret key
your_secret_key = generate_secret_key()
print("Generated Secret Key:", your_secret_key)
