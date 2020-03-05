metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
for i, metric in enumerate(metrics):
    formats = {m: "%.6f" for m in metrics}
    formats["grid_size"] = "%2d"
    formats["cls_acc"] = "%.2f%%"
    print(formats)
    #row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]