from flask import Flask, request, jsonify


class DatasetCustomNode:
    def __init__(self, data, config):
        self.data = data
        self.config = config


class DataPrepCustomNode2Sources:
    def __init__(self, data, config):
        self.data = data
        self.config = config


class DataQMLModelCustomNode:
    def __init__(self, data, config):
        self.data = data
        self.config = config


class DataOperationCustomNode:
    def __init__(self, data, config):
        self.data = data
        self.config = config


class DataOperationCustomNode2Targets:
    def __init__(self, data, config):
        self.data = data
        self.config = config


class DataOperationCustomNode:
    def __init__(self, data, config):
        self.data = data
        self.config = config


type_to_class = {
    "datasetCustomNode": DatasetCustomNode,
    "dataPrepCustomNode2Sources": DataPrepCustomNode2Sources,
    "dataQMLModelCustomNode": DataQMLModelCustomNode,
    # Add more mappings for other types...
}


# Function to create the model class based on the "type" in the JSON request
def create_model_class(request_body):
    model_instances = []
    for item in request_body:
        item_type = item.get("type")
        if item_type in type_to_class:
            model_class = type_to_class[item_type]
            model_instance = model_class(item["config"], item["data"])
            model_instances.append(model_instance)
    return model_instances


def extract_json(request):
    main_props = {}
    request_body = request.json

    if request_body is None:
        return "Invalid JSON data in request", 400

    nodes = request_body.get("nodes", [])

    # Extract keys and values of data, type, and config for each node
    node_properties = []
    for node in nodes:
        node_data = node.get("data", {})
        node_type = node.get("type")
        node_config = node.get("config", {})

        # Extract keys and values of data, config, and type
        node_data_kv = {key: node_data[key] for key in node_data}
        node_config_kv = {key: node_config[key] for key in node_config}

        node_properties.append({
            "data": node_data_kv,
            "type": node_type,
            "config": node_config_kv
        })

    # for node in node_properties:
    #     if node["type"] == "datasetCustomNode":
    #         main_props["datasets_type"] = node["data"]["label"]
        #     main_props["labels"] = node["config"]["degits_of_interest"]
        #     main_props['number_of_samples'] = node["config"]["num_samples"]
        # if node["type"] == "dataPrepCustomNode2Sources":
        #     main_props['split_ratio'] = node['config']['fraction']  #
        #     main_props['random_seed'] = node["config"]["random_seed"]
        # if node["type"] == "dataQMLModelCustomNode":
        #     main_props['qubit_number'] = node["config"]["num_qubits"]
        #     main_props['is_cuda'] = node["config"]["use_cuda"]
        #     main_props['task_name'] = node["data"]["label"]
        # if node["type"] == "dataOperationCustomNode":
        #     main_props['train_epochs'] = node["config"]["epochs"]

    objects_by_types = get_node_by_type(node_properties)
    print(objects_by_types)

    return jsonify(node_properties)


def get_node_by_type(request_body):
    object_by_type = {}

    for item in request_body:
        item_type = item.get("type")
        if item_type not in object_by_type:
            object_by_type[item_type] = []

        object_by_type[item_type].append({
            "type": item.get("type"),
            "config": item.get("config"),
            "data": item.get("data")
        })

    return object_by_type
