# 文件名: main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import optimizer

app = FastAPI(
    title="外压容器智能设计 API",
    description="提供带加强筋外压容器的全局优化设计与不加筋外压容器的最小壁厚计算两大功能。",
    version="2.0.0"
)

# === 模型定义：带筋优化 ===
class StiffenedVesselInput(BaseModel):
    length_mm: float = Field(..., title="筒体计算长度 (mm)", example=8000.0)
    diameter_mm: float = Field(..., title="筒体内径 (mm)", example=1000.0)
    temperature_c: float = Field(..., title="设计温度 (°C)", example=150.0)
    pressure_mpa: float = Field(..., title="外设计压力 (MPa)", example=1.0)

# === 模型定义：不加筋计算 ===
class UnstiffenedVesselInput(BaseModel):
    length_mm: float = Field(..., title="筒体计算长度 (mm)", example=3000.0)
    diameter_mm: float = Field(..., title="筒体内径 (mm)", example=1000.0)
    temperature_c: float = Field(..., title="设计温度 (°C)", example=30.0)
    pressure_mpa: float = Field(..., title="外设计压力 (MPa)", example=0.1)
    corrosion_allowance_mm: float = Field(2.0, title="腐蚀裕量 C1 (mm)", example=2.0)
    thickness_tolerance_mm: float = Field(0.3, title="钢板负偏差 C2 (mm)", example=0.3)


@app.post("/optimize-stiffened", summary="【优化】带加强筋容器")
def optimize_stiffened_vessel(input_data: StiffenedVesselInput):
    """
    接收带筋容器的设计参数，运行遗传算法进行全局优化，返回最优设计方案。
    """
    try:
        result = optimizer.run_api_optimization(
            length=input_data.length_mm,
            diameter=input_data.diameter_mm,
            temperature=input_data.temperature_c,
            pressure=input_data.pressure_mpa
        )
        return result
    except Exception as e:
        return {"status": "error", "message": f"优化过程中发生内部错误: {str(e)}"}


@app.post("/calculate-unstiffened", summary="【计算】不加筋容器最小壁厚")
def calculate_unstiffened_vessel(input_data: UnstiffenedVesselInput):
    """
    接收不加筋容器的设计参数，通过迭代计算，确定满足强度和稳定性要求的最小壁厚。
    """
    try:
        result = optimizer.calculate_unstiffened_thickness(
            Di=input_data.diameter_mm,
            L=input_data.length_mm,
            temp=input_data.temperature_c,
            p=input_data.pressure_mpa,
            C1=input_data.corrosion_allowance_mm,
            C2=input_data.thickness_tolerance_mm
        )
        return result
    except Exception as e:
        return {"status": "error", "message": f"计算过程中发生内部错误: {str(e)}"}

@app.get("/", summary="服务状态检查")
def read_root():
    return {"message": "欢迎使用外压容器智能设计 API v2.0，服务运行正常！"}