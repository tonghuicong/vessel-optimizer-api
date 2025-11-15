from fastapi import FastAPI
from pydantic import BaseModel
import optimizer  # 导入我们改造后的优化脚本

# 1. 创建一个 FastAPI 应用实例
app = FastAPI(
    title="外压容器优化 API",
    description="根据 GB150 标准，使用遗传算法优化带加强筋的外压容器。",
    version="1.0.0"
)

# 2. 定义输入数据的模型（确保API接收的数据格式正确）
class VesselInput(BaseModel):
    length_mm: float
    diameter_mm: float
    temperature_c: float
    pressure_mpa: float

    class Config:
        json_schema_extra = {
            "example": {
                "length_mm": 8000.0,
                "diameter_mm": 1000.0,
                "temperature_c": 150.0,
                "pressure_mpa": 1.0
            }
        }

# 3. 创建一个 API 端点 (Endpoint)
@app.post("/optimize")
async def optimize_vessel(input_data: VesselInput):
    """
    接收容器参数，运行优化，并返回最佳设计方案。
    """
    try:
        # 调用 optimizer.py 中的主函数
        result = optimizer.run_api_optimization(
            length=input_data.length_mm,
            diameter=input_data.diameter_mm,
            temperature=input_data.temperature_c,
            pressure=input_data.pressure_mpa
        )
        return result
    except Exception as e:
        # 如果出现任何错误，返回一个错误信息
        return {"status": "error", "message": f"优化过程中发生内部错误: {str(e)}"}

# 4. 根路径测试服务是否正常
@app.get("/")
def read_root():
    return {"message": "欢迎使用外压容器优化 API 服务"}
