from argparse import ArgumentParser
from os.path import join

import boto3

def get_metrics(dir_name, metric_types):
    """
    Reads the metrics files and returns a dict of metrics.
    Each key in metrics will be a dictionary with the key Value.
    For example: 'phase1_train_runtime': {'Value': 12.1279715}
    """
    metrics = {}
    for step in metric_types.keys():
        for phase in metric_types[step]['phases']:
            filename = join(dir_name, phase, metric_types[step]['filename'])
            with open(filename) as f:
                for line in f:
                    key, value = map(str.strip, line.split("="))
                    if key in metric_types[step]['keys']:
                        value = float(value)
                        # convert training runtime from seconds to hours
                        if key == 'train_runtime':
                            value = value
                            value /= 3600
                        metrics[f"{phase}_{key}"] = {
                            "Value": value,
                        }
    return metrics

def main(args):
    # the information we want to extract
    metric_types = {
        'train': {
            'filename': 'train_results.txt',
            'phases': ["bert_phase1", "bert_phase2"],
            'keys': ['train_runtime', 'train_samples_per_second'],
        },
        'squad': {
            'filename': 'eval_results.txt',
            'phases': ["squad1", "squad2"],
            'keys': ['f1'],
        }
    }

    metrics = get_metrics(args.results_dir, metric_types)
    print(metrics)

    cloudwatch = boto3.client('cloudwatch', region_name='us-west-2')

    Data = []
    namespace = f'-NLP-EC2-BERT-{args.num_nodes}nodes'

    Data = [{'MetricName': metric, 'Value': metrics[metric]['Value']} \
        for metric in metrics.keys()]

    _ = cloudwatch.put_metric_data(
        MetricData = Data,
        Namespace = namespace
    )

if __name__ == "__main__":
    parser = ArgumentParser(description='Extract metrics to be sent to CloudWatch')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory where results are')
    parser.add_argument('--num_nodes',type=int, help='Number of nodes used for training job')
    args = parser.parse_args()

    main(args)
