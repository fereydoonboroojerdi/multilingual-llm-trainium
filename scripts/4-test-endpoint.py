import boto3
client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName='multilingual-support-endpoint',
    ContentType='application/json',
    Body=json.dumps({
        'text': 'My order #12345 is delayed',
        'language': 'en',
        'domain': 'retail'
    })
)
print(json.loads(response['Body'].read()))