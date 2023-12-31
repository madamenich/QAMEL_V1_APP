from flask import Flask, request, jsonify


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

    for node in node_properties:
        if node["type"] == "datasetCustomNode":
            main_props["datasets_type"] = node.get("data", {}).get("label", "")
            main_props["labels"] = node.get("config", {}).get("degits_of_interest", [])
            main_props['number_of_samples'] = node.get("config", {}).get("num_samples", 0)
        if node["type"] == "dataPrepCustomNode2Sources":
            main_props['split_ratio'] = node.get('config', {}).get('fraction', 0.0)
            main_props['random_seed'] = node.get("config", {}).get("random_seed", 42)
        if node["type"] == "dataQMLModelCustomNode":
            main_props['qubit_number'] = node.get("config", {}).get("num_qubits", 0)
            main_props['is_cuda'] = node.get("config", {}).get("use_cuda", False)
            main_props['task_name'] = node.get("data", {}).get("label", "")
        if node["type"] == "dataOperationCustomNode":
            main_props['train_epochs'] = node.get("config", {}).get("epochs", 5)
        main_props['batch_size'] = node.get("config", {}).get("batch_size", 1)
    objects_by_types = get_node_by_type(node_properties)
    print(objects_by_types)

    return jsonify(main_props)


def extract_json2(request):
    main_props = []
    nodes = request.json
    response = {}
    training = {}
    dataset = {}
    testing = {}
    configurarion = {}

    if nodes is None:
        return "Invalid JSON data in request", 400
    for node in nodes:

        # TODO: extract the dataset
        if node.get('type') == 'dataset':
            response['datasets_type'] = node.get('subNodeType', '')
            inputs = node.get('inputs', [])
            ## TODO: handling when the user use their own dataset
            is_custom = ((response['datasets_type'] == 'csv') | (response['datasets_type'] == 'txt'))
            if is_custom:
                for input_item in inputs:
                    input_type = input_item.get('name', '')
                    if input_type == 'url':
                        response['csv_path'] = input_item.get('value', '')
                    if input_type == 'header':
                        response['header'] = input_item.get('value', '')
            else:
                for input_item in inputs:
                    input_type = input_item.get('name', '')
                    print(input_type)
                    if input_type == 'degits_of_interest':
                        response['labels'] = input_item.get('value', [])
                        print(response['labels'])
                    if input_type == 'num_samples':
                        response['number_of_samples'] = input_item.get('value', 0)
        if node.get('type') == 'dataQMLModel':
            response['task'] = node.get('subNodeType', '')
            # response['qubit_number'] = node.get('inputs')['type'] == 'qubit_number'
            inputs = node.get('inputs', [])

            for input_item in inputs:
                input_type = input_item.get('name', '')
                if input_type == 'num_qubits':
                    response['num_qubits'] = input_item.get('value', 0)
                elif input_type == 'use_cuda':
                    response['use_cuda'] = input_item.get('value', False)
                elif input_type == 'optimizer':
                    response['optimizer'] = input_item.get('value', '')
                elif input_type == 'lr':
                    response['learning_rate'] = input_item.get('value', '')

        if node.get('type') == 'dataOperation':
            inputs = node.get('inputs', [])
            for input_item in inputs:
                input_type = input_item.get('name', '')
                if input_type == 'epochs':
                    response['epochs'] = input_item.get('value', 0)
        if node.get('type') == 'dataPrep':
            inputs = node.get('inputs', [])
            for input_item in inputs:
                input_type = input_item.get('name', '')
                if input_type == 'splitting_mode':
                    response['splitting_mode'] = input_item.get('value', '')
                if input_type == 'fraction':
                    response['fraction'] = input_item.get('value', 0.7)
                if input_type == 'random_split':
                    response['random_split'] = input_item.get('value', False)
                if input_type == 'random_seed':
                    response['random_seed'] = input_item.get('value', 42)
                if input_type == 'stratify':
                    response['stratify'] = input_item.get('value', False)



    return jsonify(response)







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
