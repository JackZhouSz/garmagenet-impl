_PANEL_CLS_NUM = 18

data_fields_dict = {
    "surf_ncs": {
        "title": "Geometry Images",
        "len": 3,
    },
    "surf_wcs": {
        "title": "Geometry_Wcs Images",
        "len": 3,
    },
    "surf_uv_ncs": {
        "title": "UV Images",
        "len": 2,
    },
    "surf_normals": {
        "title": "Normals Images",
        "len": 3,
    },
    "surf_mask": {
        "title": "Mask Images",
        "len": 1,
    },
    "latent64": {
        "title": None,
        "len": 64,
    },
    "bbox3d": {
        "title": None,
        "len": 6,
    },
    "scale3d": {
        "title": None,
        "len": 3,
    },
    "bbox2d": {
        "title": None,
        "len": 4,
    },
    "scale2d": {
        "title": None,
        "len": 2,
    }
}


def get_condition_dim(args, self):
    if args.text_encoder is not None:
        condition_dim = self.text_encoder.text_emb_dim
    elif args.pointcloud_encoder is not None:
        condition_dim = self.pointcloud_encoder.pointcloud_emb_dim
    elif args.sketch_encoder is not None:
        if args.sketch_encoder == "LAION2B":
            condition_dim = 1280
        elif args.sketch_encoder == "RADIO_V2.5-G":
            condition_dim = 1536
        elif args.sketch_encoder == "RADIO_V2.5-H":
            condition_dim = 3840
        elif args.sketch_encoder == "RADIO_V2.5-H_saptial":
            condition_dim = 1280
        else:
            raise NotImplementedError("args.sketch_encoder name wrong.")
    else:
        condition_dim = -1

    return condition_dim

