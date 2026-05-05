import numpy

def analyse_results(results, o, layer):
    r = results[layer]
    vs, md = o[layer]
    output = {}
    for index, d in enumerate(r):
        vmds = d['metadatas']
        n = len(vmds)
        gt_vmd = vs.metadatas[index]
        gt_label = gt_vmd.label
        gt_efk = gt_vmd.source_echoframe_keys[0]
        first_efk = vmds[0].source_echoframe_keys[0]
        second_efk = vmds[1].source_echoframe_keys[0]
        if gt_efk != first_efk and gt_efk != second_efk:
            m = f'{index} {gt_label} {layer}'
            m += f' {gt_efk} != {first_efk}'
            m += f' and {gt_efk} != {second_efk}'
            raise ValueError(m)
        labels = [x.label for x in vmds[1:]]
        acc = labels.count(gt_label) / (n-1)
        if gt_label not in output:
            output[gt_label] = []
        output[gt_label].append(acc)
    return output

def per_word(output):
    o = {}
    for label, accs in output.items():
        o[label] = numpy.mean(accs)
    return o 

def overall_result(output):
    accs = []
    for label, acc in output.items():
        accs += acc
    return numpy.mean(accs), numpy.std(accs)


    
