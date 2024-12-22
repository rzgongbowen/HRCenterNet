import sqlite3

# 连接到数据库
conn = sqlite3.connect('database/credentials.db')
c = conn.cursor()

# 创建表
c.execute('''CREATE TABLE users
             (username text, password text)''')

# 插入示例用户
c.execute("INSERT INTO users VALUES ('root', '123')")
c.execute("INSERT INTO users VALUES ('gbw', '123')")

# 保存更改
conn.commit()

# 关闭连接
conn.close()
