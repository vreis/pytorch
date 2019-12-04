def check_if_factory_method(args):
    for arg in args: 
        if 'type' not in arg:
            return False

    a = any(arg['type'] == 'c10::optional<ScalarType>' for arg in args) and any(arg['type'] == 'c10::optional<Layout>' for arg in args) and any(arg['type'] == 'c10::optional<Device>' for arg in args) and any(arg['type'] == 'c10::optional<bool>' for arg in args)
    c = any(arg['type'] == 'ScalarType' for arg in args) and any(arg['type'] == 'Layout' for arg in args) and any(arg['type'] == 'Device' for arg in args) and any(arg['type'] == 'bool' for arg in args)
    b = any('TensorOptions' in arg['type'] for arg in args)

    return a or b or c

def collapse_actuals2(actuals):
    collapsed = actuals[:]
    index = actuals.index('dtype')
    collapsed[index] = 'at::typeMetaToScalarType(options.dtype())'
    collapsed[index + 1] = 'options.layout()'
    collapsed[index + 2] = 'options.device()'
    collapsed[index + 3] = 'options.pinned_memory()'
    return collapsed

def collapse_actuals(actuals):
        collapsed = actuals[:]
        if (any(actual == 'dtype' for actual in actuals) and
            any(actual == 'layout' for actual in actuals) and
            any(actual == 'device' for actual in actuals) and 
            any(actual == 'pin_memory' for actual in actuals)):
            index = 0
            for i in range(len(collapsed)):
                if collapsed[index] == 'dtype':
                    break
                else:
                    index += 1

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'options')

        return collapsed

def collapse_formals2(formals):
        collapsed = formals[:]
        if ((any(formal == 'c10::optional<ScalarType> dtype = c10::nullopt' for formal in formals) or any(formal == 'c10::optional<ScalarType> dtype = at::kLong' for formal in formals)) and
            any(formal == 'c10::optional<Layout> layout = c10::nullopt' for formal in formals) and
            any(formal == 'c10::optional<Device> device = c10::nullopt' for formal in formals) and 
            any(formal == 'c10::optional<bool> pin_memory = c10::nullopt' for formal in formals)):
            if 'c10::optional<ScalarType> dtype = c10::nullopt' in formals:
                index = formals.index('c10::optional<ScalarType> dtype = c10::nullopt')
            else:
                index = formals.index('c10::optional<ScalarType> dtype = at::kLong')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const at::TensorOptions & options={}')

        if (any(formal == 'at::ScalarType dtype' for formal in formals) and
            any(formal == 'at::Layout layout' for formal in formals) and
            any(formal == 'at::Device device' for formal in formals) and 
            (any(formal == 'bool pin_memory' for formal in formals) or any(formal == 'bool pin_memory = false' for formal in formals))):
            index = formals.index('at::ScalarType dtype')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const at::TensorOptions & options')
        
        return collapsed

def collapse_formals(formals):
        collapsed = formals[:]
        if (any(formal == 'c10::optional<ScalarType> dtype' for formal in formals) and
            any(formal == 'c10::optional<Layout> layout' for formal in formals) and
            any(formal == 'c10::optional<Device> device' for formal in formals) and 
            any(formal == 'c10::optional<bool> pin_memory' for formal in formals)):
            index = formals.index('c10::optional<ScalarType> dtype')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const TensorOptions & options /*[CHECK THIS] should have ={}*/')

        if (any(formal == 'c10::optional<ScalarType> dtype=c10::nullopt' for formal in formals) and
            any(formal == 'c10::optional<Layout> layout=c10::nullopt' for formal in formals) and
            any(formal == 'c10::optional<Device> device=c10::nullopt' for formal in formals) and 
            (any(formal == 'c10::optional<bool> pin_memory=c10::nullopt' for formal in formals) or any(formal == 'c10::optional<bool> pin_memory=false' for formal in formals))):
            index = formals.index('c10::optional<ScalarType> dtype=c10::nullopt')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const TensorOptions & options={}')

        return collapsed

def collapse_formals_list(formals):
        collapsed = formals[:]
        if (any(formal['type'] == 'c10::optional<ScalarType>' for formal in collapsed) and 
            any(formal['type'] == 'c10::optional<Layout>' for formal in collapsed) and 
            any(formal['type'] == 'c10::optional<Device>' for formal in collapsed) and 
            any(formal['type'] == 'c10::optional<bool>' for formal in collapsed)):
            index = 0
            for i in range(len(collapsed)):
                if collapsed[i]['type'] == 'c10::optional<ScalarType>':
                    break
                else:
                    index += 1

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, {"annotation" : "None", "dynamic_type": "TensorOptions", "is_nullable": "False", "default": "{}", "kwarg_only": "True", "name": "options", "type": "const TensorOptions &", })

        if (any(formal['type'] == 'ScalarType' for formal in collapsed) and 
            any(formal['type'] == 'Layout' for formal in collapsed) and 
            any(formal['type'] == 'Device' for formal in collapsed) and 
            any(formal['type'] == 'bool' for formal in collapsed)):
            index = 0
            for i in range(len(collapsed)):
                if collapsed[i]['type'] == 'ScalarType':
                    break
                else:
                    index += 1

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, {"annotation" : "None", "dynamic_type": "TensorOptions", "is_nullable": "False", "kwarg_only": "True", "name": "options", "type": "const TensorOptions &", })

        return collapsed

def check_tensor_options_in_formals(formals):
    return (any(formal['dynamic_type'] == 'ScalarType' for formal in formals) and
            any(formal['dynamic_type'] == 'Layout' for formal in formals) and
            any(formal['dynamic_type'] == 'Device' for formal in formals) and 
            any(formal['dynamic_type'] == 'bool' for formal in formals))