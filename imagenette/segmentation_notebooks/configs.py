unet_config = {
    'inputs_shapes': (3, 256, 256),
    
    'order': ['encoder', 'embedding', 'decoder', 'head'],
    

    'encoder': { 'type': 'encoder',
                 'order': ['block', 'skip', 'downsample'],
                 'num_stages': 4,
                 'blocks': { 'layout': 'cna cna',
                             'kernel_size':3,
                             'padding': 1,
                             'stride': 1,
                             'channels': [64, 128, 256, 512],
                             'bias': False
                            },
                'downsample': {'layout': 'p',
                               'kernel_size': 2,
                               'stride': 2,
                               'channels': 'same * 2'
                              } 
               },
    'embedding': {
                'input_type': 'list',
                'input_idx': -1,
                'output_type': 'list',
                'layout': 'cna cna',
                'kernel_size': 3,
                'padding': 1,
                'channels': 1024,
            },
    
    'decoder': {
        'skip': True,
        'indices':  [3, 2, 1, 0],
        'type': 'decoder',
        'num_stages': 4,
        'order': ['upsample', 'combine', 'block'],
        'blocks': {'layout': 'cna cna', 
                   'channels': [512, 256, 128, 64],
                   'kernel_size': 3, 
                   'stride': 1,
                   'padding': 1,
                   'bias': False
                  },
        'combine': {'op': 'concat', 'force_resize': False}, 
        'upsample': {'layout': 't', 'kernel_size': 2, 'stride': 2, 'channels': 'same // 2', 'bias': True}
    },
    'head': {
        'layout': 'c', 'channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True
    },
    
    'loss': 'ce', 
    'device': 'gpu:0',
}
