import timm

all_models = timm.list_models()
vit_models = [model for model in all_models if 'vit' in model]

print(f'Total number of ViT models: {len(vit_models)}')
print('Available ViT models in timm:')
for model in vit_models:
    print(model)