import matplotlib.pyplot as plt

"""# Dữ liệu
learning_rates = [ 0.0001, 0.001]
mAP_scores = [ 0.0000, 0.0120]

# Vẽ đồ thị
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, mAP_scores, marker='o', linestyle='-', color='blue', label='mAP vs Learning Rate')

# Đặt log scale cho trục x để dễ nhìn hơn
plt.xscale('log')

# Thêm tiêu đề và nhãn trục
plt.title('Biểu đồ mAP theo Learning Rate')
plt.xlabel('Learning Rate ')
plt.ylabel('mAP')

# Hiển thị giá trị tại từng điểm
for x, y in zip(learning_rates, mAP_scores):
    plt.text(x, y + 0.01, f"{y:.5f}", ha='center')

# Thêm lưới và chú thích
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""


models = ['YOLOv11', 'RT-DETR', 'Faster R-CNN']
map_scores = [0.64, 12.30, 146.14]  # mAP@0.5 

# === Vẽ biểu đồ cột ===
plt.figure(figsize=(8, 6))
bars = plt.bar(models, map_scores, color=['skyblue', 'salmon', 'lightgreen'])

# Thêm giá trị mAP trên đầu mỗi cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

plt.title("Flops Comparison Across Models")
plt.xlabel("Model")
plt.ylabel("Flops (GFLOPs)")
plt.ylim(0, 150)  # mAP từ 0 đến 1
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()