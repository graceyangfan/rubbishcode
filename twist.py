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
    BiType,
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
    bi_type = BiType.OLD 
    fx_included = False 
    zs_type = ZSType.INSIDE
    zs_support_type = SupportType.HL 
    macd_fast = 12 
    macd_slow = 26
    macd_signal = 9 
    boll_period = 20 
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
        self.ma_period = config.ma_period 

        self.macd = MovingAverageConvergenceDivergence(
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        self.boll = BollingerBands(self.boll_period)
        self.ma = ExponentialMovingAverage(self.ma_period) 

        # 计算后保存的值
        self.newbars = deque(maxlen = config.bar_capacity)          # 整理后的原始K线
        self.twist_bars = deque(maxlen=config.twist_bar_capacity)   # 缠论K线
        self.fxs = deque(maxlen=config.fx_capacity)                 # 分型列表
        self.bis = deque(maxlen=config.bi_capacity)
        self.xds = deque(maxlen=config.xd_capacity)
        self.big_trends = deque(maxlen=config.trend_capacity)
        self.bi_zss = deque(maxlen=config.bi_zs_capacity)
        self.xd_zss = deque(maxlen=config.xd_zs_capacity) 
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
            bar_type = bar.bar_type,
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
                new_bar.ts_opened,
                new_bar.ts_closed,
                [new_bar],
                new_bar.open,
                new_bar.high,
                new_bar.low,
                new_bar.close,
                new_bar.volume,
                False,
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

        post_twist_bars = self.process_twist_bar(raw_bars,up_twist_bars) 
        if twisit_bar_2:
            self.twist_bars.pop() 
            self.twist_bars.pop() 
        else:
            self.twist_bars.pop() 
        ##add processed twist_bars 
        for item in post_twist_bars:
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
            if self.fake_bi:
                b1,b2 = self.twist_bars[-2:]
                b3 = None 
                if b2.high > b1.high:
                    fx = FX(
                        mark_type = Mark.DING,
                        middle_twist_bar = b2,
                        twist_bars = [b1,b2,b3],
                        value = b2.high,
                        jump = False,
                        real = True,
                        is_confirm= False,  
                    )
                elif b2.low < b1.low:
                    fx = FX(
                        mark_type = Mark.DI,
                        middle_twist_bar = b2,
                        twist_bars = [b1,b2,b3],
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
            self.fxs.append(fx)
            return True    

        ##check if fx should be updated 
        is_update = False  
        end_fx = self.fxs[-1]
        if fx.ts_opened == end_fx.ts_opened:
            self.fxs[-1] = fx 
            is_update = True 
        
        
        up_fx = None
        # record max and min value in un_real fxs 
        fx_interval_high = None
        fx_interval_low = None
        for _fx in np.array(self.fxs)[::-1]:
            if is_update and _fx.ts_opened == fx.ts_opened:
                continue 
            fx_interval_high = _fx.value if fx_interval_high is None else max(fx_interval_high, _fx.value) 
            fx_rang_low = _fx.value if fx_interval_low is None else min(fx_interval_low,_fx.value) 
            if _fx.real:
                up_fx = _fx 
                break 
        
        if up_fx is None:
            return False

        if self.bi_type == BiType.TB:
            if not is_update:
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
                or fx.middle_twist_bar.low <= up_fx.middle_twost_bar.high 
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
            if self.bi_type == BiType.OLD and fx.middle_twist_bar.index - up_fx.middle_twist_bar.index < 4:
                fx.real = False 
            if self.bi_type == BiType.NEW and (fx.middle_twist_bar.index - up_fx.middle_twist_bar.index < 3 \
                or fx.middle_twist_bar.elements[-1].index - up_fx.middle_twist_bar.elements[-1].index < 4):
                fx.real = False 
        if is_update is False:
            self.fxs.append(fx)

        ## clear un_real fx 
        for _fx in self.fxs:
            if not _fx.real:
                self.fxs.remove(_fx)
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
            if bi.is_confirm and bi.direction_type == Direction.UP and close < bi.end.twistbars[-1].low:
                bi.pause = True 
            elif bi.is_confirm and bi.direction_type == Direction.DOWN and close > bi.end.twistbars[-1].high:
                bi.pause = True 
            else:
                bi.pause = False 
        if not bi: ## the first time to generate bi 
            if len(self.fxs) < 2:
                return False 
            for fx in self.fxs:
                if bi is None:
                    bi = BI(start = fx) 
                    continue 
                if bi.start.mark_type == fx.mark_type:
                    continue 
                bi.end = fx 
                bi.direction_type = Direction.UP if bi.start.type == Mark.DI else Direction.DOWN 
                bi.is_confirm = True 
                bi.jump = False 
                self.process_line_power(bi)
                self.process_line_hl(bi)
                self.bis.append(bi) 
                return True 

        ## decide the last fx 
        end_real_fx = self.fxs[-1] 
        
        if bi.end.ts_opened == end_real_fx.ts_opened and bi.is_confirm != end_real_fx.is_confirm:
            bi.end = end_real_fx 
            bi.is_confirm = end_real_fx.confirm 
            self.process_line_power(bi)
            self.process_line_hl(bi)
            return True 

        if bi.end.ts_opened < end_real_fx.ts_opened and bi.end.mark_type != end_real_fx.mark_type:
            # new bi generate 
            new_bi = BI(start=bi.end, end=end_real_fx)
            new_bi.direction_type = Direction.UP if new_bi.start.mark_type == Mark.DI else Direction.DOWN 
            new_bi.is_confirm = end_real_fx.is_confirm 
            new_bi.pause  = False 
            self.process_line_power(bi)
            self.process_line_hl(bi)
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
            up_lines = list(self.xds)
            base_lines = list(self.bis)
        elif base_line_type == LineType.XD:
            up_lines = list(self.big_trends) 
            base_lines = list(self.xds)
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
                    start_line= start,
                    end_line = end_line,
                    direction_type = Direction.UP if end_fx.makr_type == Mark.DING else Direction.DOWN,
                    ding_fx = start_fx if start_fx.mark_type == Mark.DING else end_fx,
                    di_fx = start_fx if start_fx.mark_type == Mark.DI else end_fx,
                    is_confirm= end_fx.confirm,
                )
                self.process_line_power(new_up_line)
                self.process_line_hl(new_up_line)
                up_lines.append(new_up_line)
                return True
            else:
                return False


        ## generally update XD 
        up_line = up_lines[-1]
        if up_line.direction_type == Direction.UP:
            dings = self.cal_line_xlfx(base_lines[up_line.start_line.index:],Mark.DING) 
            for ding in dings:
                if ding.line.index >=up_line.end_line.index:
                    end_line = base_lines[ding.line.index - 1]
                    up_lines.end = end_line.end 
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


    def process_zs(
        self,
        run_types = None,
    ):
        """
        generate zss.
        """
        if run_types is None:
            return False
        for run_type in run_types:
            if run_type == LineType.BI:
                lines = list(self.bis)
                up_lines = list(self.xds) 
                zss = list(self.bi_zss) 
            elif run_type == LineType.XD:
                lines = list(self.xds)
                up_lines = list(self.big_trends)
                zss = list(self.xd_zss) 
            else:
                raise Exception('error zs run_type as %s' % run_type)

            if len(lines) < 4:
                return False
            if self.zs_type == ZSType.INTERATE:
                self.process_interate_zs(lines, zss, run_type)
            elif self.zs_type == ZSType.INSIDE:
                self.process_inside_zs(lines, up_lines, zss, run_type)
            else:
                raise Exception('error zs run_type as %s' % run_type)

        return True


    def process_inside_zs(
        self, 
        lines: List[LINE], 
        up_lines: List[XD], 
        zss: List[ZS], 
        run_type: ZSType,
    ):
        """
        compute dn zs 
        """
        if len(up_lines) >= 2:
            _up_l = up_lines[-2]
            _run_lines = lines[_up_l.start_line.index:_up_l.end_line.index + 1]
            _up_zss = [_zs for _zs in zss if _up_l.start.index <= _zs.start.index < _up_l.end.index]
            _new_zss = self.create_dn_zs(run_type, _run_lines)
            for _u_zs in _up_zss:
                if _u_zs.start.index not in [_z.start.index for _z in _new_zss]:
                    _u_zs.real = False
                    continue
                for _n_zs in _new_zss:
                    if _u_zs.start.index == _n_zs.start.index:
                        self.__copy_zs(_n_zs, _u_zs)
                        _u_zs.is_confirm = True

        # compute dn zs 
        run_lines: List[LINE]
        if len(up_lines) == 0:
            run_lines = lines
        else:
            run_lines = lines[up_lines[-1].start_line.index:]

        exists_zs = [_zs for _zs in zss if _zs.start.index >= run_lines[0].start.index]
        new_zs = self.create_dn_zs(run_type, run_lines)
        #update or remove zs 
        for _ex_zs in exists_zs:
            if _ex_zs.start.index not in [_z.start.index for _z in new_zs]:
                _ex_zs.real = False
                continue
            for _n_zs in new_zs:
                if _n_zs.start.index == _ex_zs.start.index:
                    self.__copy_zs(_n_zs, _ex_zs)
        # add new zs
        for _n_zs in new_zs:
            if _n_zs.start.index not in [_z.start.index for _z in exists_zs]:
                _n_zs.index = zss[-1].index + 1 if len(zss) > 0 else 0
                zss.append(_n_zs)
        return

    def process_interate_zs(
        self,
        lines: List[LINE], 
        zss: List[ZS], 
        run_type: ZSType,
    ):
        """
        compute BL ZS.
        """

        if len(zss) == 0:
            _ls = lines[-4:]
            _zs = self.create_zs(run_type, None, _ls)
            if _zs:
                zss.append(_zs)
            return True

        line = lines[-1]
        # get  unconfirm zs to recompute with bis 
        for _zs in zss:
            ## confirmed zs 
            if _zs.is_confirm:
                continue
            if _zs.end.index == line.end.index:
                continue
            # update _zs property with list 
            self.create_zs(run_type, _zs, lines[_zs.lines[0].index:line.index + 1])
            # if cureent bi is in future of zs's finial bi,zs is confirmed 
            if line.index - _zs.lines[-1].index > 1:
                _zs.is_confirm = True
                if len(_zs.lines) < 5:  # len_bis small than 5 is not a ZS
                    _zs.real = False

        # create zs with last four bis 
        _zs = self.create_zs(run_type, None, lines[-4:])
        if _zs:
            # check if zs is exists 
            is_exists = False
            for __zs in zss[::-1]:
                if __zs.start.index == _zs.start.index:
                    is_exists = True
                    break
            if is_exists is False:
                _zs.index = zss[-1].index + 1
                zss.append(_zs)
        return True


    def process_trade_point(
        self, 
        run_types=None
        ):
        """
        compute Divergence and TradePoints
        """
        if run_types is None:
            return False

        for run_type in run_types:
            if run_type == LineType.BI:
                lines: List[BI] = self.bis
                zss: List[ZS] = self.bi_zss
            elif run_type == LineType.XD:
                lines: List[XD] = self.xds
                zss: List[ZS] = self.xd_zss
            else:
                raise Exception('trade point based line_type error ：%s' % run_type)

            if len(zss) == 0:
                return True

            line = lines[-1]
            # clear trade points and recompute 
            line.bcs = []
            line.trade_points = []

            # add bi divergence 
            line.add_bc(run_type, None, lines[-3], self.divergence_line(lines[-3], line))
            # find all zss end with current line 
            line_zss: List[ZS] = [
                _zs for _zs in zss
                if (_zs.lines[-1].index == line.index and _zs.real and _zs.level == 0)
            ]
            for _zs in line_zss:
                line.add_bc('pz', _zs, _zs.lines[0], self.divergence_pz(_zs, line))
                line.add_bc('qs', _zs, _zs.lines[0], self.divergence_trend(zss, _zs, line))

            #  trend divergence (1buy,1sell)
            for bc in line.bcs:
                if bc.type == 'qs' and bc.bc:
                    if line.direction_type == Direction.UP:
                        line.add_trade_point('1sell', bc.zs)
                    if line.direction_type == Direction.DOWN:
                        line.add_trade_point('1buy', bc.zs)

            # 2buy,2sell，同向的前一笔突破，再次回拉不破，或者背驰，即为二类买卖点
            for _zs in line_zss:
                if len(_zs.lines) < 7:
                    continue
                tx_line: [BI, XD] = _zs.lines[-3]
                if _zs.lines[0].direction_type == Direction.UP and line.direction_type == Direction.UP:
                    if tx_line.high == _zs.gg and (tx_line.high > line.high or line.bc_exists(['pz', 'qs'])):
                        line.add_trade_point('2sell', _zs)
                if _zs.lines[0].direction_type == Direction.DOWN and line.direction_type == Direction.DOWN:
                    if tx_line.low == _zs.dd and (tx_line.low < line.low or line.bc_exists(['pz', 'qs'])):
                        line.add_trade_point('2buy', _zs)

            # l2buy,l2sell, When first bir is (2buy,2sell) and leave power is weaker than back power
            for _zs in line_zss:
                # if ZS has invere trade_points or divergence trade_point,then no l2 trade_points 
                have_buy = False
                have_sell = False
                have_bc = False
                for _line in _zs.lines[:-1]:
                    if _line.trade_point_exists(['1buy', '2buy', 'l2buy', '3buy', 'l3buy']):
                        have_buy = True
                    if _line.trade_point_exists(['1sell', '2sell', 'l2sell', '3sell', 'l3sell']):
                        have_sell = True
                    if _line.bc_exists(['pz', 'qs']):
                        have_bc = True
                if '2buy' in _zs.lines[1].line_trade_points() and line.direction_type == Direction.DOWN:
                    if have_sell is False and have_bc is False and self.compare_power_divergence(_zs.lines[1].power, line.power):
                        line.add_trade_point('l2buy', _zs)
                if '2sell' in _zs.lines[1].line_trade_points() and line.direction_type == Direction.UP:
                    if have_buy is False and have_bc is False and self.compare_power_divergence(_zs.lines[1].power, line.power):
                        line.add_trade_point('l2sell', _zs)

            # 3buy,3sell, ZS's end line is just the previous of the last bi.
            line_3trade_point_zss: List[ZS] = [
                _zs for _zs in zss
                if (_zs.lines[-1].index == line.index - 1 and _zs.real and _zs.level == 0)
            ]
            for _zs in line_3trade_point_zss:
                if len(_zs.lines) < 5:
                    continue
                if line.direction_type == Direction.UP and line.high < _zs.zd:
                    line.add_trade_point('3sell', _zs)
                if line.direction_type == Direction.DOWN and line.low > _zs.zg:
                    line.add_trade_point('3buy', _zs)

            # l3buy,l3sell trade point 
            for _zs in line_zss:
                # if ZS has invere trade_points or divergence trade_point,then no l3 trade_points 
                have_buy = False
                have_sell = False
                have_bc = False
                for _line in _zs.lines[:-1]:
                    # 不包括当前笔
                    if _line.trade_point_exists(['1buy', '2buy', 'l2buy', '3buy', 'l3buy']):
                        have_buy = True
                    if _line.trade_point_exists(['1sell', '2sell', 'l2sell', '3sell', 'l3sell']):
                        have_sell = True
                    if _line.bc_exists(['pz', 'qs']):
                        have_bc = True
                for trade_point in _zs.lines[1].trade_points:
                    if trade_point.name == '3buy':
                        if have_sell is False and have_bc is False and line.direction_type == Direction.DOWN \
                                and line.low > trade_point.zs.zg \
                                and self.compare_power_divergence(_zs.lines[0].power, line.power):
                            line.add_trade_point('l3buy', trade_point.zs)
                    if trade_point.name == '3sell':
                        if have_buy is False and have_bc is False and line.direction_type == Direction.UP \
                                and line.high < trade_point.zs.zd \
                                and self.compare_power_divergence(_zs.lines[0].power, line.power):
                            line.add_trade_point('l3sell', trade_point.zs)

        return True


    def process_line_power(self, line: LINE):
        """
        process Line power 
        """
        line.power = {
            'macd': self.query_macd_ld(line.start, line.end)
        }
        return True 

    def line_hl(
        self,
        line: LINE
    ):
        if self.zs_support_type == SupportType.HL:
            return [line.high,line.low]
        else:
            return [line.ding_high(),line.di_low()]

    def process_line_hl(self, line: LINE):
        """
        process line real high low point 
        """
        fx_bars = self.newbars[line.start.middle_twist_bar.index:line.end.middle_twist_bar.index+1]
        high_array = [item.high for item in fx_bars]
        low_array = [item.low for item in fx_bars] 
        return max(high_array),min(low_array) 

    def create_zs(
        self, 
        zs_type: ZSType, 
        zs: [ZS, None], 
        lines: List[LINE],
    ) -> [ZS, None]:
        """
        if pass zs:update its property 
        if not pass zs:create a new zs 
        """
        if len(lines) <= 3:
            return None
    

        run_lines = []
        zs_confirm  = False

        ## record overlap of zs 
        _high,_low = self.line_hl(lines[0]) 

        interval_high, interval_low = self.cross_interval(
            self.line_hl(lines[1]),
            self.line_hl(lines[3])
        )

        if interval_high is None:
            return None 
   
        for _l in lines:
            #get all lines overlap to zs,once not overlap the zs has finished.
            _l_hl = self.line_hl(_l)
            if self.cross_interval(
                [interval_high,interval_low],
                _l_hl,
            ):
                _high = max(_high, _l_hl[0]) 
                _low = min(_low, _l_hl[1])
                run_lines.append(_l)
            else:
                zs_confirm = True 
                break 
        if len(run_lines) < 4:
            return None 

        _last_line = run_lines[-1] 
        _last_hl = self.line_hl(_last_line) 
        last_line_in_zs = True 
        if (_last_line.direction_type == Direction.UP and _last_hl[0] == _high) \
            or (_last_line.direction_type == Direction.DOWN and _last_hl[1] == _low):
            #if the last line is the highest or lowest,it do not belong to zs 
            last_line_in_zs = False 

        if zs is None:
            zs = ZS(
                zs_type = zs_type,
                start = run_lines[1].start,
                direction_type = Direction.OSCILLATION,
            )
        zs.is_confirm = zs_confirm 
   
        zs.lines = []
        zs.add_line(run_lines[0])
        zs_range = [interval_high, interval_low ]
        zs_gg = run_lines[1].high 
        zs_dd = run_lines[1].low 

        for i in range(1,len(run_lines)):
            _l = run_lines[i] 
            _l_hl = self.line_hl(_l) 
            cross_range = self.cross_interval(zs_range,_l_hl)
            if not cross_range:
                raise Exception("A ZS must have overlap interval")

            if i == len(run_lines) - 1 and last_line_in_zs is False:
                # the last line and it is not incuded in zs 
                pass
            else:
                zs_gg = max(zs_gg, _l_hl[0])
                zs_dd = min(zs_dd, _l_hl[1])
                # compute level based on line_num 
                zs.line_num = len(zs.lines) - 1
                zs.level = int(zs.line_num / 9)
                zs.end = _l.end
                # record max power 
                if zs.max_power is None:
                    zs.max_power = _l.power 
                elif _l.power:
                    zs.max_power = zs.max_power if self.compare_power_divergence(zs.max_power, _l.power) else _l.power
            zs.add_line(_l)

        zs.zg = zs_range[0]
        zs.zd = zs_range[1]
        zs.gg = zs_gg
        zs.dd = zs_dd

        # compute zs direction 

        if zs.lines[0].type == zs.lines[-1].type:
            _l_start_hl = self.line_hl(zs.lines[0])
            _l_end_hl = self.line_hl(zs.lines[-1])
            if zs.lines[0].direction_type == Direction.UP and _l_start_hl[1] <= zs.dd and _l_end_hl[0] >= zs.gg:
                zs.direction_type= zs.lines[0].direction_type
            elif zs.lines[0].direction_type == Direction.DOWN and _l_start_hl[0] >= zs.gg and _l_end_hl[1] <= zs.dd:
                zs.direction_type = zs.lines[0].direction_type
            else:
                zs.direction_type = Direction.OSCILLATION
        else:
            zs.direction_type = Direction.OSCILLATION

        return zs

    def create_dn_zs(
        self, 
        zs_type: ZSType,
        lines: List[LINE]
    ) -> List[ZS]:
        """
        compute dn zs.
        """
        zss = [] 
        if len(lines) <= 4:
            return zss

        start = 0
        while True:
            run_lines = lines[start:]
            if len(run_lines) == 0:
                break
            zs = self.create_zs(zs_type, None, run_lines)
            if zs is None:
                start += 1
            else:
                zss.append(zs)
                start += len(zs.lines) - 1

        return zss



    def divergence_line(
        self, 
        pre_line: LINE, 
        now_line: LINE
    ):
        """
        compare if there has a divergence between two lines 
        """
        if pre_line.direction_type != now_line.direction_type:
            return False 
        if pre_line.direction_type == Direction.UP and now_line.high < pre_line.high:
            return False 
        if pre_line.direction_type == Direction.DOWN and now_line.low > pre_line.low:
            return False 

        return self.compare_power_divergence(pre_line.power, now_line.power)

    def divergence_pz(self, zs: ZS, now_line: LINE):
        """
        decied if the ZS is in pz divergence 
        """
        if zs.lines[-1].index != now_line.index:
            return False
        if zs.zs_direction_type not in [Direction.UP,Direction.DOWN]:
            return False 
 
        return self.compare_power_divergence(zs.lines[0].power, now_line.power)

    def divergence_trend(self, zss: List[ZS], zs: ZS, now_line: LINE):
        """
        decide  if the ZS us in trend divergence 
        """
        if zs.zs_direction_type not in [Direction.UP,Direction.DOWN]:
            return False 

        # check if has same diretion ZS 
        pre_zs = [
            _zs for _zs in zss
            if (_zs.lines[-1].index == zs.lines[0].index and _zs.direction_type == zs.direction_type and _zs.level == zs.level)
        ]
        if len(pre_zs) == 0:
            return False
        # if high and low overlaped 
        pre_overlap_zs = []
        for _zs in pre_zs:
            if (_zs.direction_type == Direction.UP and _zs.gg < zs.dd) or (_zs.direction_type == Direction.DOWN and _zs.dd > zs.gg):
                pre_overlap_zs.append(_zs)

        if len(pre_overlap_zs) == 0:
            return False

        return self.compare_power_divergence(zs.lines[0].power, now_line.power)


    @staticmethod
    def cross_interval(interval_one, interval_two):
        """
        compute the cross of two interval 
        :param interval_one:
        :param interval_two:
        :return:
        """

        max_one = max(interval_one[0], interval_one[1])
        min_one = min(interval_one[0], interval_one[1])
        max_two = max(interval_two[0], interval_two[1])
        min_two = min(interval_two[0], interval_two[1])

        cross_max_val = min(max_two, max_one)
        cross_min_val = max(min_two, min_one)

        if cross_max_val >= cross_min_val:
            return  cross_max_val,cross_min_val
        else:
            return None,None 

    @staticmethod
    def compare_power_divergence(one_power: dict, two_power: dict):
        """
        compute  if there has a divergence with macd value sum.
        """
        hist_key = 'sum'

        if two_power['macd']['hist'][hist_key] < one_power['macd']['hist'][hist_key]:
            return True
        else:
            return False


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
                newbars[0].ts_opened,
                newbars[0].ts_closed,
                [newbars[0]],
                newbars[0].open,
                newbars[0].high,
                newbars[0].low,
                newbars[0].close,
                newbars[0].volume,
                newbars[0].jump,
            )
        twist_bars.append(twist_bar)
        up_twist_bars.append(twist_bar)

        for i in range(1,len(newbars)):
            twist_b = twist_bars[-1]
            newbar = newbars[i] 
            if (twist_b.high >= newbar.high and twist_b.low <= newbar.low) or (twist_b.high <= newbar.high and twist_b.low >= newbar.low):
                ## direct aggregate 
                if len(up_twist_bars) < 2:
                    twist_b.high = max(twist_b.high, newbar.high) 
                    twist_b.low = min(twist_b.low, newbar.low)
                else: 
                    #up direction 
                    if up_twist_bars[-2].high < twist_b.high:
                        twist_b.high = max(twist_b.high, newbar.high) 
                        twist_b.low = max(twist_b.low, newbar.low)
                        twist_b.previous_trend = Direction.UP 
                    else:
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
                    newbar.ts_opened,
                    newbar.ts_closed,
                    [newbar],
                    newbar.open,
                    newbar.high,
                    newbar.low,
                    newbar.close,
                    newbar.volume,
                    newbar.jump,
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
            if (fx_type == Mark.DING and line.direction_type == Direction.DOWN) or (fx_type == Mark.DI and line.direction_type == Mark.UP):
                now_xl = TZXL(
                        high = line.ding_high(),
                        low = line.di_low(),
                        line = line,
                        line_broken= False,
                )
                if len(sequence) == 0:
                    sequence.append(now_xl)

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
                elif up_xl.high < now_xl.high and up_xl.low > now_xl.now:
                    #strong included ,current xl include front xl
                    now_xl.line_broken = True 
                    sequence.append(now_xl)
                else:
                    sequence.append(now_xl)
        xlfxs = [] 
        for i in range(1,len(sequence)):
            up_xl = sequence[i-1]
            now_xl = sequence[i] 
            if len(sequence) > (i+1):
                next_xl = sequence[i+1]
            else:
                next_xl = None 

            jump = True if up_xl.high < now_xl.min or up_xl.low > now_xl.high else False 

            if next_xl:
                fx_high = max(up_xl.high,now_xl.high,next_xl.high)
                fx_low = min(up_xl.low,now_xl.low,next_xl.low)

                if fx_type == Mark.DING and up_xl.high < now_xl.high and now_xl.high > next_xl.high:
                    now_xl.mark_type = Mark.DING 
                    xlfxs.append(
                        XLFX(
                            mark_type =Mark.DING,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl,
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
                            line = now_xl,
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
                            line = now_xl,
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
                            line = now_xl,
                            jump =jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = False,
                        )
                    )

        return xlfxs

    def on_reset(self):
        """
        Actions to be performed when the strategy is reset.
        """
        # Reset indicators here
        self.macd.reset()
        self.boll.reset() 
        self.ma.reset() 
