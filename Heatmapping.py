import polars as pl
import numpy as np
import matplotlib.pyplot as plt

def draw_mapping(x, y, values, grid_size=1, cmap='Reds', 
                 title='Value Mapping', colorlable='Value', vmin=0, vmax=2,
                 invert_xaxis=False, invert_yaxis=False):  # 新增反转参数
    """
    创建二维坐标数值分布热力图
    
    参数:
    -----------
    x : array-like
        X坐标数组
    y : array-like
        Y坐标数组
    values : array-like
        每个坐标点的数值
    grid_size : float, 可选
        网格大小 (默认: 1)
    cmap : str, 可选
        颜色映射 (默认: 'coolwarm')
    title : str, 可选
        图表标题 (默认: '数值分布热力图')
    colorlable : str, 可选
        颜色条标签 (默认: 'Value')
    vmin, vmax : float, 可选
        颜色映射的最小值和最大值
    invert_xaxis : bool, 可选
        是否反转X轴 (默认: False)
    invert_yaxis : bool, 可选
        是否反转Y轴 (默认: False)
    """
    x_min, y_min = np.min(x), np.min(y)
    
    x_idx = ((np.array(x) - x_min) / grid_size).astype(int)
    y_idx = ((np.array(y) - y_min) / grid_size).astype(int)
    
    grid_shape = (x_idx.max() + 1, y_idx.max() + 1)
    grid = np.full(grid_shape, np.nan)
    
    for xi, yi, val in zip(x_idx, y_idx, values):
        grid[xi, yi] = val
    
    plt.figure(figsize=(8, 6))
    

    pc = plt.pcolormesh(
        np.arange(x_min, x_min + grid_shape[0] * grid_size, grid_size),
        np.arange(y_min, y_min + grid_shape[1] * grid_size, grid_size), 
        grid.T, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        shading='auto')
    
    # 坐标轴反转设置
    if invert_xaxis:
        plt.gca().invert_xaxis()
    if invert_yaxis:
        plt.gca().invert_yaxis()
    
    plt.colorbar(pc, label=colorlable)
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 读取数据
    df = pl.read_csv('MergedData_490R620_1.csv')
    df = df.select(['position_x', 'position_y', 'predict_wld'])
    df = df.to_pandas()
    
    x = df['position_x'].to_numpy().astype(int)
    y = df['position_y'].to_numpy().astype(int)
    values = df['predict_wld'].to_numpy()

    # 绘制热力图（添加反转参数）
    draw_mapping(
        x,
        y,
        values,
        cmap='coolwarm',
        title='Heatmap',
        colorlable='predict_wld',
        vmin=622.5,
        vmax=625,
        invert_xaxis=False,  # 反转X轴
        invert_yaxis=False   # 反转Y轴
    )