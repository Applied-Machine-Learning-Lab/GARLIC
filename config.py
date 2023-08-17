import argparse
import os

from utils.parser_utils import add_flags_from_config

config_args = {
    'model_config': {
        'model': ('GreedyClosest', 'which pretrain_model to use: [GreedyClosest, ]'),
        'isshared':(False, 'ride sharing or not'),
        'graph-model':('GAT', 'model for graph embedding [GAT, GCN]'),
        'node_classes':(3, '0 for demand < supply, 1 for demand = supply, 2 for demand > supply'),
        'graph_dim': (32, 'embedding dimension'),
        'dim': (128, 'embedding dimension'),
        'act': ('gelu', 'which activation function to use (or None for no activation)'),
        'act_gpt':('gelu_python', ''),
        'num-layers': (3, 'number of hidden layers in encoder, at least 2'),
        'dropout': (0.5, 'dropout probability'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'warm-steps':(1000,''),
    },
    'training_config':{
        'epochs': (10000, 'maximum number of epochs to train for'),
        'update-freq':(5, 'number of epochs update reward function & graph embedding'),
        'batch-size':(64, 'batch size'),
        'balance':(True, 'Ensure supply and demand balance'),
        'balance-window':(120, 'balance task window'),
        'predict-window':(600, 'prediction window'),
        'step':(5, 'time steps for one calculation'),
        'cuda':(0, '-1 for using CPU'),
        'log-dir':('logs', 'logging directory'),
        'lr': (0.001, 'learning rate'),
        'weight-decay': (0., 'l2 regularization strength'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'grad-clip': (5, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'max-len':(50, 'maximum token length'),
        'reward-gamma':(0.999, '')
    },
    'environment_config':{
        'start-time': ("17:00:00", 'data start time'),
        'end-time': ("21:00:00", 'data end time'),
        'taxi-size':(500, 'number of taxis')
    },
    'sumo_config': {
        'nogui':(False, 'run the commandline version of sumo'),
        'sim-time':(15000, 'total simulation step'),
        'sim-file':("data/sumo_files/sumo.sumocfg", 'sumo simulation file'),
        'pre-sim-file':("data/sumo_files/sumo_preprocessing.sumocfg", ''),
        'sumo-gui-setting':('data/sumo_files/gui-settings.xml',''),
        'net-file': ("data/sumo_files/convert.net.xml", 'road network dir & file'),
        'add-file':('data/sumo_files/type.add.xml', 'additional files'),
        'edge-pair-time':("data/preprocess/time_cost.pkl", 'travel time for edge pairs'),
        'high-light':(40, 'high light size'),
        'err-log':('logs/errors.txt', 'error information'),
        'pair-err-log':('logs/pair_error.txt', 'error information'),
        'DS-window':(600, 'demand supply aggregate window'),
        'pred-time':(600,'prediction time'),
        'rebalance-time':(600,''),
        'rebalance-task-time':(120, '')
    },
    'data_config':{
        'north-lat':(30.390, 'north boundary of the road network'),
        'south-lat':(30.133, 'south boundary of the road network'),
        'east-lon':(120.301, 'south boundary of the road network'),
        'west-lon':(120.071, 'south boundary of the road network'),
        'hg2_radius':(1000, 'meter'),
        'hg5_radius':(2000, 'meter'),
        'hg10_radius':(5000, 'meter'),
        'city':('hangzhou', ''),
        'hg2-zones-shp':("data/taxi_zones/hg2/hg2.shp",''),
        'hg5-zones-shp':("data/taxi_zones/hg5/hg5.shp",''),
        'hg10-zones-shp':("data/taxi_zones/hg10/hg10.shp",''),
        'hg2-central-edge':("data/taxi_zones/hg2/centralEdges.json", 'zone central edge file'),
        'hg5-central-edge':("data/taxi_zones/hg5/centralEdges.json", 'zone central edge file'),
        'hg10-central-edge':("data/taxi_zones/hg10/centralEdges.json", 'zone central edge file'),
        'zone-edge':("data/taxi_zones/zone_edges.csv", 'zone central edge file'),
        'hg2-relation':("data/taxi_zones/hg2/zone_relation.csv", 'zone relation file'),
        'hg5-relation':("data/taxi_zones/hg5/zone_relation.csv", 'zone relation file'),
        'hg10-relation':("data/taxi_zones/hg10/zone_relation.csv", 'zone relation file'),
        'road-centerline':("data/road_centerline/road_centerline.osm", ''),
        'person-demand-before':('data/sumo_files/convert-demandtrips.xml',''),
        'person-demand-after':('data/sumo_files/demand.xml',''),
        'ori-data':("data/preprocess/demand_trips.csv", ''),
        'tmp-time-change-data':("data/preprocess/tmp_demand_trips.csv", ''),
        'tmp-edge-done-data':("data/preprocess/tmp_edgedone.csv", ''),
        'tmp-person-data':('data/preprocess/24hour_end.csv', ''),
        'tmp-person-data2':('data/preprocess/24hour_end_1.csv', ''),
        'tmp-curr-routes':('data/sumo_files/out/curr_routes.rou.xml', 'in controller._predictFutureSupply()'),
        'tmp-res-routes':('data/sumo_files/out/resultroutes.rou.xml', ''),
        'trip-info':("data/sumo_files/out/tripinfos.xml", 'trip info output file'),
        'stop-info':("data/sumo_files/out/stopinfo.xml", 'stop info output file'),
        'vehicle-route':('data/sumo_files/out/vehroute.rou.xml', 'vehicle route output file'),
        'tmp-demand-graph':('data/preprocess/tmp_demand_graph.csv', ''),
        'trajectory-pos':('data/hangzhou/trajectory_pos.pkl', ''),
        'trajectory-neg':('data/hangzhou/trajectory_neg.pkl', ''),
        'trajectory-all':('data/hangzhou/trajectory_all.pkl', ''),
        'multigraph-adj':('data/hangzhou/adjlist.pkl', ''),
        'multigraph-fea':('data/hangzhou/fealist.pkl', ''),
        'max-car-num':(50, '')
    },
    'sql_config':{
        'host':('127.0.0.1', 'mysql host'),
        'port':(3306, 'mysql port'),
        'usr':('root', 'mysql user'),
        'passwd':('123456', 'mysql user password'),
        'db':('mytest', 'mysql database name'),
        'table':('taxi_run_20170930', 'mysql data table')
    }
}


parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)