# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2022 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------


from base_object import (
    BI,
    NewBar,
    TwistBar,
    FX,
    LINE,
    ZS,
    TradePoint,
    BC,
    TZXL,
    XLFX,
    XD
)
import numpy as np
from enums import (
    Mark,
    Direction,
    BIType,
    LineType,
    ZSType,
    SupportType
)
from collections import deque 
from typing import List, Dict
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.bollinger_bands import BollingerBands


class TwistConfig(StrategyConfig):
    bar_type: str
    instrument_id: str
    fake_bi = False  # if recognize un_confirmed bi 
    bi_type = BIType.OLD 
    fx_included = False 
    zs_type = ZSType.INSIDE
    zs_support_type = SupportType.HL 
    macd_fast = 12 
    macd_slow = 26
    macd_signal = 9 
    boll_period = 20 
    boll_k = 2 
    ma_period = 5 
    bar_capacity: int = 1000
    twist_bar_capacity: int = 1000
    fx_capacity: int = 500
    bi_capacity: int = 500
    xd_capacity: int = 500
    trend_capacity:int = 500
    bi_zs_capacity: int = 250  
    xd_zs_capacity: int = 250 


    ##不继承Strategy 
class Twist:
    """
    行情数据缠论分析
    """

    def __init__(self, config: TwistConfig):
        """
        """
        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.fake_bi = config.fake_bi 
        self.bi_type =config.bi_type 
        self.fx_included = config.fx_included 
        self.zs_type = config.zs_type 
        self.zs_support_type = config.zs_support_type 

        self.macd_fast = config.macd_fast 
        self.macd_slow = config.macd_slow 
        self.macd_signal = config.macd_signal 
        self.boll_period = config.boll_period 
        self.boll_k = config.boll_k
        self.ma_period = config.ma_period 

        self.macd = MovingAverageConvergenceDivergence(
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        self.boll = BollingerBands(self.boll_period,self.boll_k)
        self.ma = ExponentialMovingAverage(self.ma_period) 

        # 计算后保存的值
        self.newbars = []       # 整理后的原始K线
        self.twist_bars = []  # 缠论K线
        self.fxs = []             # 分型列表 
        self.real_fxs = []   
        self.bis = [] 
        self.xds = []
        self.big_trends = []
        self.bi_zss = []
        self.xd_zss = []
        self.bar_count= 0 

    def on_start(self,strategy: Strategy):
        """
        call from outside stratgies 
        """
        self.instrument = strategy.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        # Register the indicators for updating
        strategy.register_indicator_for_bars(self.bar_type, self.macd)
        strategy.register_indicator_for_bars(self.bar_type, self.boll) 
        strategy.register_indicator_for_bars(self.bar_type, self.ma)


    def on_bar(self,bar: Bar):
        ## update newbar 
        info = self.get_newbar_info() 
        new_bar = NewBar(
            bar_type = self.bar_type,
            index = self.bar_count,
            ts_opened = bar.ts_event,
            ts_closed =  bar.ts_event,
            open = bar.open.as_double(),
            high = bar.high.as_double(),
            low = bar.low.as_double(),
            close = bar.close.as_double(),
            volume = bar.volume.as_double(),
            info = info,
        )
        self.newbars.append(new_bar) 
        ##update twist bars 
        self.process_twist_bar() 
        ##update fx 
        self.process_fx() 
        ##update bi 
        bi_update = self.process_bi()

        ##update xd from bi 
        xd_update = self.process_up_line(LineType.BI) if bi_update else False
        #self.process_up_line(LineType.XD) if xd_update else False

        ##update bar_count 
        self.bar_count += 1

    def get_newbar_info(self):
        return None 

    def process_twist_bar(self):
        """
        Aggregate ordinary bars in twist bars.
        """
        new_bar = self.newbars[-1] 
        if len(self.twist_bars) < 1:
            twist_bar = TwistBar(
                index = 0,
                ts_opened = new_bar.ts_opened,
                ts_closed = new_bar.ts_closed,
                elements= [new_bar],
                open = new_bar.open,
                high = new_bar.high,
                low = new_bar.low,
                close = new_bar.close,
                volume = new_bar.volume,
                jump = False,
            )
            self.twist_bars.append(twist_bar)
            return True 
        if len(self.twist_bars) >=4:
            up_twist_bars = [self.twist_bars[-4],self.twist_bars[-3]]
        elif len(self.twist_bars) <=2:
            up_twist_bars = [] 
        else:
            up_twist_bars =[self.twist_bars[-3]] 
        twisit_bar_1 = self.twist_bars[-1]
        twisit_bar_2 = self.twist_bars[-2] if len(self.twist_bars) >=2 else None 

        raw_bars = twisit_bar_2.elements if twisit_bar_2  else [] 
        raw_bars.extend(twisit_bar_1.elements) 
        if new_bar.ts_opened != raw_bars[-1].ts_opened:
            raw_bars.append(new_bar) 

        post_twist_bars = self.bars_inlcuded(raw_bars,up_twist_bars) 
        if twisit_bar_2:
            self.twist_bars.pop() 
            self.twist_bars.pop() 
        else:
            self.twist_bars.pop() 
        ##add processed twist_bars 
        for item in post_twist_bars:
            if len(self.twist_bars) < 1:
                item.index = 0 
            else:
                item.index = self.twist_bars[-1].index + 1 
            self.twist_bars.append(item) 
        return True 

    def process_fx(self):
        """
        Aggregate twist bars into FX.
        """
        if len(self.twist_bars) < 3:
            return False 
        
        b1, b2, b3 = self.twist_bars[-3:]
        fx = None 
        if (b1.high < b2.high and b2.high > b3.high) and (b1.low < b2.low and b2.low > b3.low):
            jump = True if (b1.high < b2.low or b2.low > b3.high) else False 
            fx = FX(
                mark_type = Mark.DING,
                middle_twist_bar = b2,
                twist_bars = [b1,b2,b3],
                value = b2.high,
                jump = jump,
                real = True,
                is_confirm= True, 
            )
        if (b1.high > b2.high and b2.high < b3.high) and (b1.low > b2.low and b2.low < b3.low):
            jump = True  if (b1.low > b2.high or b2.high < b3.low) else False
            fx = FX(
                mark_type = Mark.DI,
                middle_twist_bar = b2,
                twist_bars = [b1,b2,b3],
                value = b2.low,
                jump = jump,
                real = True,
                is_confirm= True, 
            )
        
        if fx is None:
            ##check un_confirmed FX 
            if self.fake_bi:
                b1,b2 = self.twist_bars[-2:]
                b3 = None 
                if b2.high > b1.high:
                    fx = FX(
                        mark_type = Mark.DING,
                        middle_twist_bar = b2,
                        twist_bars = [b1,b2],
                        value = b2.high,
                        jump = False,
                        real = True,
                        is_confirm= False,  
                    )
                elif b2.low < b1.low:
                    fx = FX(
                        mark_type = Mark.DI,
                        middle_twist_bar = b2,
                        twist_bars = [b1,b2],
                        value = b2.low,
                        jump = False,
                        real = True,
                        is_confirm= False,  
                    )
                else:
                    return False 
            else:
                return False 

        if len(self.fxs) == 0 and fx.is_confirm is False:
            return False
        elif len(self.fxs) == 0 and fx.is_confirm is True:
            fx.index = 0 
            self.fxs.append(fx)
            return True    

        ##check if fx should be updated 
        is_update = False  
        end_fx = self.fxs[-1]
        if fx.ts_opened == end_fx.ts_opened:
            end_fx_index = end_fx.index
            self.fxs[-1] = fx 
            self.fxs[-1].index = end_fx_index
            is_update = True 
        
        
        up_fx = None
        # record max and min value in un_real fxs 
        fx_interval_high = None
        fx_interval_low = None
        for _fx in self.fxs[::-1]:
            if is_update and _fx.ts_opened == fx.ts_opened:
                continue 
            fx_interval_high = _fx.value if fx_interval_high is None else max(fx_interval_high, _fx.value) 
            fx_rang_low = _fx.value if fx_interval_low is None else min(fx_interval_low,_fx.value) 
            if _fx.real:
                up_fx = _fx 
                break 
        
        if up_fx is None:
            return False

        if self.bi_type == BIType.TB:
            if not is_update:
                fx.index = self.fxs[-1].index + 1 
                self.fxs.append(fx) 
            return True 

        
        if fx.mark_type == Mark.DING and up_fx.mark_type == Mark.DING and up_fx.middle_twist_bar.high <= fx.middle_twist_bar.high:
            up_fx.real = False 
        elif fx.mark_type == Mark.DI and up_fx.mark_type == Mark.DI and up_fx.middle_twist_bar.low >= fx.middle_twist_bar.low:
            up_fx.real = False
        elif fx.mark_type == up_fx.mark_type:
            fx.real = False  ## continue fx, DING - prev_high > last_high ,drop last 
        elif fx.mark_type == Mark.DING and up_fx.mark_type == Mark.DI and \
            (
                fx.middle_twist_bar.high <= up_fx.middle_twist_bar.low 
                or fx.middle_twist_bar.low <= up_fx.middle_twist_bar.high 
                or (not self.fx_included  and fx.high() < up_fx.high())
        ):
            fx.real = False 
        elif fx.mark_type == Mark.DI and up_fx.mark_type == Mark.DING and \
            (
                fx.middle_twist_bar.low >= up_fx.middle_twist_bar.high 
                or fx.middle_twist_bar.high >= up_fx.middle_twist_bar.low 
                or (not self.fx_included and fx.low() > up_fx.low())
        ):
            fx.real = False 
        else:
            if self.bi_type == BIType.OLD and fx.middle_twist_bar.index - up_fx.middle_twist_bar.index < 4:
                fx.real = False 
            if self.bi_type == BIType.NEW and (fx.middle_twist_bar.index - up_fx.middle_twist_bar.index < 3 \
                or fx.middle_twist_bar.elements[-1].index - up_fx.middle_twist_bar.elements[-1].index < 4):
                fx.real = False 
        if not is_update:
            fx.index = self.fxs[-1].index + 1 
            self.fxs.append(fx)

        ##get_real_fxs 
        if len(self.fxs) >= 2:
            if not self.fxs[-2].real:
                if not self.fxs[-1].real:
                    self.fxs.pop()
                    self.fxs.pop() 
                else:
                    self.fxs[-2] = self.fxs[-1]
                    self.fxs[-2].index = self.fxs[-3].index + 1  if len(self.fxs) >=3 else 0 
                    self.fxs.pop()
            else:
                if not self.fxs[-1].real:
                    self.fxs.pop() 
        elif len(self.fxs) == 1:
            if not self.fxs[-1].real:
                self.fxs.pop()
        else:
            pass 
        '''
        self.fxs = [_fx for _fx in self.fxs if _fx.real]
        for i in range(len(self.fxs)):
            self.fxs[i].index = self.fxs[0].index + 1
        '''
        return True
        
    def process_bi(self):
        """
        Aggregate FXs into bis.
        """
        if len(self.fxs) == 0:
            return False


        if len(self.bis) > 0 and  not self.bis[-1].start.real:
            self.bis.pop()
        
        bi = self.bis[-1] if len(self.bis) > 0 else None 
        ##check bi  pause 
        if bi:
            close = self.newbars[-1].close 
            if bi.is_confirm and bi.direction_type == Direction.UP and close < bi.end.twist_bars[-1].low:
                bi.pause = True 
            elif bi.is_confirm and bi.direction_type == Direction.DOWN and close > bi.end.twist_bars[-1].high:
                bi.pause = True 
            else:
                bi.pause = False 

        if bi is None: ## the first time to generate bi 
            if len(self.fxs) < 2:
                return False 
            for fx in self.fxs:
                if bi is None:
                    bi = BI(start = fx, index = 0) 
                    continue 
                if bi.start.mark_type == fx.mark_type:
                    continue 
                bi.end = fx 
                bi.direction_type = Direction.UP if bi.start.mark_type == Mark.DI else Direction.DOWN 
                bi.is_confirm = fx.is_confirm  
                bi.jump = False 
                self.process_line_power(bi)
                self.process_line_hl(bi)
                self.bis.append(bi)
                return True 

        ## decide the last fx 
        end_real_fx = self.fxs[-1]
        if (bi.end.real is False and bi.end.mark_type == end_real_fx.mark_type) or \
            (bi.end.index == end_real_fx.index and bi.is_confirm != end_real_fx.is_confirm):
            bi.end = end_real_fx 
            bi.is_confirm = end_real_fx.is_confirm 
            self.process_line_power(bi)
            self.process_line_hl(bi)
            return True 

        if bi.end.index < end_real_fx.index and bi.end.mark_type != end_real_fx.mark_type:
            # new bi generate 
            new_bi = BI(start=bi.end, end=end_real_fx)
            new_bi.index = self.bis[-1].index + 1
            new_bi.direction_type = Direction.UP if new_bi.start.mark_type == Mark.DI else Direction.DOWN 
            new_bi.is_confirm = end_real_fx.is_confirm 
            new_bi.pause  = False 
            self.process_line_power(new_bi)
            self.process_line_hl(new_bi)
            self.bis.append(new_bi)
            return True 

        return False

    def process_up_line(
        self,
        base_line_type = LineType.BI,
    ):
        """
        Aggregate bis into XLFX and XD.
        """
        is_update = False
        if base_line_type == LineType.BI:
            up_lines = self.xds
            base_lines = self.bis
        elif base_line_type == LineType.XD:
            up_lines = self.big_trends
            base_lines = self.xds
        else:
            raise ('high level xd name is wrong：%s' % base_line_type)

        if len(base_lines) == 0:
            return False
        ##first time update XD 
        if len(up_lines) == 0:
            bi_0 = base_lines[0] 
            start_fx = XLFX(
                mark_type= Mark.DI if bi_0.direction_type == Direction.UP else Mark.DING,
                high = bi_0.high,
                low = bi_0.low,
                line = bi_0, 
            )
            end_fx = None 
            if start_fx.mark_type == Mark.DI:
                dis = self.cal_line_xlfx(base_lines, Mark.DI)
                for di in dis:
                    if di.line.index > start_fx.line.index: 
                        start_fx = di 
                dings = self.cal_line_xlfx(base_lines[start_fx.line.index:], Mark.DING) 
                for ding in dings:
                    if ding.line.index - start_fx.line.index >= 2:
                        ## general new XD 
                        end_fx = ding
                        break
            elif start_fx.mark_type == Mark.DING:
                dings = self.cal_line_xlfx(base_lines, Mark.DING)
                for ding in dings:
                    if ding.line.index > start_fx.line.index:
                        start_fx = ding 
                dis  = self.cal_line_xlfx(base_lines[start_fx.line.index:], Mark.DI)
                for di in dis:
                    if di.line.index - start_fx.line.index >=2:
                        end_fx = di 

            if start_fx and end_fx:
                start_line = start_fx.line 
                end_line = base_lines[end_fx.line.index-1]
                new_up_line = XD(
                    start = start_line.start,
                    end = end_line.end,
                    start_line= start_line,
                    end_line = end_line,
                    direction_type = Direction.UP if end_fx.mark_type == Mark.DING else Direction.DOWN,
                    ding_fx = start_fx if start_fx.mark_type == Mark.DING else end_fx,
                    di_fx = start_fx if start_fx.mark_type == Mark.DI else end_fx,
                    is_confirm= end_fx.is_confirm,
                )
                self.process_line_power(new_up_line)
                self.process_line_hl(new_up_line)
                up_lines.append(new_up_line)
                return True
            else:
                return False


        ## generally update XD 
        up_line = up_lines[-1]
        ## if extened 
        if up_line.direction_type == Direction.UP:
            dings = self.cal_line_xlfx(base_lines[up_line.start_line.index:],Mark.DING) 
            for ding in dings:
                if ding.line.index >=up_line.end_line.index:
                    end_line = base_lines[ding.line.index - 1]
                    up_line.end = end_line.end 
                    up_line.end_line = end_line 
                    up_line.ding_fx = ding 
                    up_line.is_confirm = ding.is_confirm 
                    self.process_line_power(up_line)
                    self.process_line_hl(up_line)
                    is_update = True 
        elif up_line.direction_type == Direction.DOWN:
            dis = self.cal_line_xlfx(base_lines[up_line.start_line.index:], Mark.DI)
            for di in dis:
                if di.line.index >= up_line.end_line.index:
                    end_line = base_lines[di.line.index - 1]
                    up_line.end = end_line.end
                    up_line.end_line = end_line 
                    up_line.di_fx = di 
                    up_line.is_confirm = di.is_confirm 
                    self.process_line_power(up_line)
                    self.process_line_hl(up_line)
                    is_update = True

        ##check if has inverse-direction XLFX to generate new XD 
        if up_line.direction_type == Direction.UP:
            dis = self.cal_line_xlfx(base_lines[up_line.end_line.index+1:],Mark.DI)
            for di in dis:
                if di.line.index - up_line.end_line.index >=2 :
                    start_line = base_lines[up_line.end_line.index+1]
                    end_line = base_lines[di.line.index-1]
                    new_up_line = XD(
                        start= start_line.start,
                        end = end_line.end,
                        start_line= start_line,
                        end_line = end_line,
                        direction_type= Direction.DOWN,
                        ding_fx= up_line.ding_fx,
                        di_fx = di,
                        index = up_line.index + 1,
                        is_confirm= di.is_confirm,
                    )
                    self.process_line_power(new_up_line)
                    self.process_line_hl(new_up_line)
                    # two DD uncomplete 
                    up_line.is_confirm = True 
                    up_lines.append(new_up_line)
                    is_update = True
                    break
        elif up_line.direction_type == Direction.DOWN:
            dings = self.cal_line_xlfx(base_lines[up_line.end_line.index + 1:], Mark.DING)
            for ding in dings:
                if ding.line.index - up_line.end_line.index >= 2: 
                    start_line = base_lines[up_line.end_line.index + 1]
                    end_line = base_lines[ding.line.index - 1]
                    new_up_line = XD(
                        start=start_line.start,
                        end=end_line.end,
                        start_line=start_line,
                        end_line=end_line,
                        direction_type= Direction.UP,
                        ding_fx=ding,
                        di_fx=up_line.di_fx,
                        index = up_line.index + 1,
                        is_confirm= ding.is_confirm,
                    )
                    self.process_line_power(new_up_line)
                    self.process_line_hl(new_up_line)
                    ## two DD uncomplete 
                    up_line.is_confirm = True
                    up_lines.append(new_up_line)
                    is_update = True
                    break

        return is_update


    def process_line_power(self, line: LINE):
        """
        process Line power 
        """
        line.power = {
            'macd': self.query_macd_power(line.start, line.end)
        }
        return True 

    def line_hl(
        self,
        line: LINE
    ):
        if self.zs_support_type == SupportType.HL:
            return [line.high,line.low]
        else:
            return [line.top_high(),line.bottom_low()]

    def process_line_hl(self, line: LINE):
        """
        process line real high low point 
        """
        fx_bars = self.newbars[line.start.middle_twist_bar.index:line.end.middle_twist_bar.index+1]
        b_h = [_b.high for _b in fx_bars]
        b_l = [_b.low for _b in fx_bars]
        line.high = np.array(b_h).max()
        line.low = np.array(b_l).min()
        return True 

    def query_macd_power(self, start_fx: FX, end_fx: FX):
        if start_fx.ts_opened > end_fx.ts_opened:
            raise Exception("start_fx start time should small than end_fx's start time ")
        return None 

    @staticmethod
    def bars_inlcuded(
        newbars: List[NewBar], 
        up_twist_bars: List[TwistBar]
    ) -> List[TwistBar]:
        """
        Aggregate ordinary bars in twist bars.

        Parameters
        ----------
        newbars : List[NewBar]
            The original bars.
        up_twist_bars : List[TwistBar]
            The Third and fourth twist bars away from current time.
        """
        twist_bars = [] 
        twist_bar = TwistBar(
                index = newbars[0].index, ##will be replaced,no worry 
                ts_opened = newbars[0].ts_opened,
                ts_closed = newbars[0].ts_closed,
                elements = [newbars[0]],
                open = newbars[0].open,
                high = newbars[0].high,
                low = newbars[0].low,
                close = newbars[0].close,
                volume = newbars[0].volume,
                jump = False,
            )
        twist_bars.append(twist_bar)
        up_twist_bars.append(twist_bar)

        for i in range(1,len(newbars)):
            twist_b = twist_bars[-1]
            newbar = newbars[i] 
            if (twist_b.high >= newbar.high and twist_b.low <= newbar.low) or (twist_b.high <= newbar.high and twist_b.low >= newbar.low):
                ## direct aggregate 
                if len(up_twist_bars) < 2:
                    #twist_b.index = twist_b.index 
                    twist_b.high = max(twist_b.high, newbar.high) 
                    twist_b.low = min(twist_b.low, newbar.low)
                else: 
                    #up direction 
                    if up_twist_bars[-2].high < twist_b.high:
                        twist_b.index = twist_b.index if twist_b.high > newbar.high else newbar.index 
                        twist_b.high = max(twist_b.high, newbar.high) 
                        twist_b.low = max(twist_b.low, newbar.low)
                        twist_b.previous_trend = Direction.UP 
                    else:
                        twist_b.index = twist_b.index if twist_b.low < newbar.low else newbar.index 
                        twist_b.high = min(twist_b.high, newbar.high) 
                        twist_b.low = min(twist_b.low, newbar.low)
                        twist_b.previous_trend = Direction.DOWN 

                twist_b.ts_opened = twist_b.ts_opened
                twist_b.ts_closed = newbar.ts_closed 
                twist_b.open = twist_b.open 
                twist_b.close = newbar.close 
                twist_b.volume = twist_b.volume + newbar.volume 
                twist_b.elements.append(newbar) 
            else:
                twist_bar = TwistBar(
                    index= newbar.index,
                    ts_opened = newbar.ts_opened,
                    ts_closed = newbar.ts_closed,
                    elements = [newbar],
                    open = newbar.open,
                    high = newbar.high,
                    low = newbar.low,
                    close = newbar.close,
                    volume = newbar.volume,
                    jump = False,
                )
                twist_bars.append(twist_bar)
                up_twist_bars.append(twist_bar) 
                
        return  twist_bars 

    @staticmethod
    def cal_line_xlfx(
        lines: List[LINE],
        fx_type= Mark.DING,
    ) -> List[XLFX]:
        """
        use line high low point two compute XLFXS
        """
        sequence= []
        for line in lines:
            if (fx_type == Mark.DING and line.direction_type == Direction.DOWN) or (fx_type == Mark.DI and line.direction_type == Direction.UP):
                now_xl = TZXL(
                        high = line.top_high(),
                        low = line.bottom_low(),
                        line = line,
                        line_broken= False,
                )
                if len(sequence) == 0:
                    sequence.append(now_xl)
                    continue 

                trend = Direction.UP if fx_type == Mark.DING else Direction.DOWN 
                up_xl = sequence[-1] 

                if up_xl.high > now_xl.high and up_xl.low <=now_xl.low:
                    if trend == Direction.UP:
                        now_xl.line = now_xl.line if now_xl.high >= up_xl.high else up_xl.line 
                        now_xl.high = max(up_xl.high,now_xl.high) 
                        now_xl.low = max(up_xl.low,now_xl.low)
                    else:
                        now_xl.line = now_xl.line if now_xl.low <= up_xl.low else up_xl.line 
                        now_xl.high = min(up_xl.high,now_xl.high) 
                        now_xl.low = min(up_xl.low,now_xl.low) 
                    sequence.pop() 
                    sequence.append(now_xl) 
                elif up_xl.high < now_xl.high and up_xl.low > now_xl.low:
                    #strong included ,current xl include front xl
                    now_xl.line_broken = True 
                    sequence.append(now_xl)
                else:
                    sequence.append(now_xl)
        xlfxs: List[XLFX] = [] 
        for i in range(1,len(sequence)):
            up_xl = sequence[i-1]
            now_xl = sequence[i] 
            if len(sequence) > (i+1):
                next_xl = sequence[i+1]
            else:
                next_xl = None 

            jump = True if up_xl.high < now_xl.low or up_xl.low > now_xl.high else False 

            if next_xl:
                fx_high = max(up_xl.high, now_xl.high, next_xl.high)
                fx_low = min(up_xl.low, now_xl.low, next_xl.low)

                if fx_type == Mark.DING and up_xl.high < now_xl.high and now_xl.high > next_xl.high:
                    now_xl.mark_type = Mark.DING 
                    xlfxs.append(
                        XLFX(
                            mark_type =Mark.DING,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump = jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = True,
                        )
                    )
                if fx_type == Mark.DI and up_xl.low > now_xl.low and now_xl.low < next_xl.low:
                    now_xl.mark_type = Mark.DI
                    xlfxs.append(
                        XLFX(
                            mark_type = Mark.DI,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump =jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = True,
                        )
                    )
            else:
                ##uncomplete FX 
                fx_high = max(up_xl.high,now_xl.high)
                fx_low = min(up_xl.low,now_xl.low)

                if fx_type == Mark.DING and up_xl.high < now_xl.high:
                    now_xl.mark_type = Mark.DING 
                    xlfxs.append(
                        XLFX(
                            mark_type =Mark.DING,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump = jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = False,
                        )
                    )
                if fx_type == Mark.DI and up_xl.low > now_xl.low:
                    now_xl.mark_type = Mark.DI
                    xlfxs.append(
                        XLFX(
                            mark_type = Mark.DI,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump =jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = False,
                        )
                    )

        return xlfxs