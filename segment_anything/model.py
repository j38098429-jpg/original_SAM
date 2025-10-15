from segment_anything import sam_model_registry

    
def build_model(args, device):
    model,_ = sam_model_registry[args.vit_type](args, image_size=args.img_size,
                                                    num_classes=args.num_classes,
                                                    chunk = 15,#args.max_timeframe,  # default, ignore
                                                    checkpoint=args.resume, pixel_mean=[0., 0., 0.],)

    return model.to(device=device)

