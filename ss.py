from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

# 读取训练图像
datagen = N2V_DataGenerator()

imgs = datagen.load_imgs_from_directory(
    directory="train_images",
    dims="YX"
)

# 从含噪图像中裁剪训练 patch
X = datagen.generate_patches_from_list(
    imgs,
    shape=(64, 64)
)

# 配置模型
config = N2VConfig(
    X,
    unet_kern_size=3,
    train_steps_per_epoch=100,
    train_epochs=100,
    train_loss="mse",
    batch_norm=True,
    train_batch_size=16,
    n2v_perc_pix=0.198,
    n2v_patch_shape=(64, 64),
    n2v_manipulator="uniform_withCP",
    n2v_neighborhood_radius=5
)

# 创建并训练模型
model = N2V(
    config=config,
    name="noise2void_model",
    basedir="models"
)

history = model.train(X, X)