
import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
import jax
import openpi.training.sharding as sharding


def main():
    
    config = _config.get_config("pi05_tron_single_data_lora")
    
    mesh = sharding.make_mesh(config.fsdp_devices)
    
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,           # 数据自动分片到各设备
        shuffle=True,                     # 打乱数据
    )
    
    data_iter = iter(data_loader)
    batch = next(data_iter)

    print(batch[1])
    


if __name__=="__main__":
    main()