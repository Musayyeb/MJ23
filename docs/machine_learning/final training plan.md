

Manumap     6100

auto hus9:  7600

auto hus1:  6700


manu 6100 => training  manu_8128 20 iterations Span=10 allspec batch=10  epochs=40


now with a better mapping algorithm !

predict hus9 => lmap_9    ==> new automap_9 data for training

predict hus1 => lmap_1    ==> new automap_1 data for training

get automap data only from healthy blocks and best ratings


limit automap to 10000 letters (or 100000?)



ML_Central  creates 1 Model  span? batch? epochs? layer?  => with iterations
            (no reduce)

input data = manumap and/or automap_1  and/or  automap_9  ???
sizes:       6100           20000              20000            ==> 46000
sizes:       6100           50000              50000            ==> 106000
sizes:       6100          100000             100000            ==> 200000

span ==> 0 / 5 (*3) / 10 (*5)
