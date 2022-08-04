from twist_add import Twist,TwistConfig 
from nautilus_trader.config.backtest import  BacktestDataConfig
import os 
if __name__ == "__main__":
    config = TwistConfig(
        instrument_id="AVAXBUSD.BINANCE",
        bar_type="AVAXBUSD.BINANCE-1-MINUTE-LAST-EXTERNAL",
    )
    cl = Twist(config)
    CATALOG_PATH =  "catalog"
    data_config = BacktestDataConfig(
            catalog_path=str(CATALOG_PATH),
            data_cls="nautilus_trader.model.data.bar:Bar",
            instrument_id="AVAXBUSD.BINANCE",
        )
    bar_data = data_config.load()["data"]
    print(len(bar_data))
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    for bar in bar_data:
        cl.on_bar(bar)
    print(len(cl.fxs))
    profiler.stop()

    profiler.print()
    print(len(cl.bis))
    print(cl.fxs[0].middle_twist_bar.elements[0].index)
    for item in cl.bis:
        print(item.index)

    ##梳理Line 和BC 