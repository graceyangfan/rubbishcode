from twist import Twist,TwistConfig 

if __name__ == "__main__":
    config = TwistConfig(
        instrument_id="GALBUSD-PERP.BINANCE",
        bar_type="GALBUSD-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL",
    )
    cl = Twist(config)
