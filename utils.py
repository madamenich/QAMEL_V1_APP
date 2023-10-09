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


# Mapping of type to class
NODE_TYPE_MAP = {
    "datasetCustomNode": DatasetCustomNode,
    "dataPrepCustomNode2Sources": DataPrepCustomNode2Sources,
    "dataQMLModelCustomNode": DataQMLModelCustomNode,
    "dataOperationCustomNode": DataOperationCustomNode,
    "dataOperationCustomNode2Targets": DataOperationCustomNode2Targets,
    "dataOperationCustomNode": DataOperationCustomNode
}

# Your JSON data
json_data = [...]  # List of dictionaries, each representing a node

# Create instances based on type


def extract_properties(request):
    json_data = extract_json(request)
    node_instances = []
    for node_data in json_data:
        node_type = node_data["type"]
        data = node_data.get("data", {})
        config = node_data.get("config", {})

        if node_type in NODE_TYPE_MAP:
            node_instance = NODE_TYPE_MAP[node_type](data, config)
            node_instances.append(node_instance)

    # Example usage
    for node_instance in node_instances:
        print(type(node_instance).__name__)
        print("Data:", node_instance.data)
        print("Config:", node_instance.config)
        print()


def extract_json(request) :
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

    return jsonify(node_properties)