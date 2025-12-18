import pyodbc

# MDB 文件路径
mdb_file = r"C:\Users\Administrator\Desktop\RNA\RLaffinity-main\3dcnn_lba\data\train\data.mdb"

# 连接字符串（Microsoft Access Driver）
conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    rf"DBQ={mdb_file};"
)

try:
    # 建立连接
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # 查询示例
    table_name = "YourTableName"  # 替换为实际表名
    cursor.execute(f"SELECT * FROM [{table_name}]")

    # 打印数据
    columns = [desc[0] for desc in cursor.description]
    print(" | ".join(columns))
    for row in cursor.fetchall():
        print(" | ".join(str(x) for x in row))

    # 关闭连接
    cursor.close()
    conn.close()

except Exception as e:
    print("连接或读取失败:", e)
