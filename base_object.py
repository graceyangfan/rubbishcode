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

from nautilus_trader.core.data import Data
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.data.bar import Bar,BarType 
import math 

'''
Basic  objects for twist compute 
'''

class NewBar:
    def __init__(
        self,
        bar_type,
        index,
        ts_opened,
        ts_closed,
        open,
        high,
        low,
        close,
        volume,
        info,
    ):
        self.bar_type = bar_type 
        self.ts_opened = ts_opened 
        self.ts_closed = ts_closed 
        self.index = index 
        self.open = open 
        self.high = high 
        self.low = low 
        self.close = close 
        self.volume = volume 
        self.info = info 

class TwistBar:
    def __init__(
        self,
        index,
        ts_opened,
        ts_closed,
        elements,
        open,
        high,
        low,
        close,
        volume,
        jump = False, #gap  #是否有缺口
    ):
        self.index = index 
        self.ts_opened = ts_opened 
        self.ts_closed = ts_closed 
        self.elements = elements 
        self.open = open 
        self.high = high 
        self.low = low 
        self.close = close 
        self.volume = volume 
        self.jump = jump 
        self.previous_trend = None  

    
    def raw_bars(self):
        return self.elements 

    def bar_type(self):
        return self.elements[0].bar_type 


class FX:
    def __init__(
        self,
        mark_type, # Mark.DING  
        middle_twist_bar,
        twist_bars,
        value,
        jump,#跳空 
        real,
        is_confirm,
    ):
        self.mark_type = mark_type 
        self.middle_twist_bar = middle_twist_bar
        self.twist_bars = twist_bars 
        self.value = value 
        self.jump = jump  
        self.real = real 
        self.is_confirm = is_confirm 
        self.ts_opened = self.twist_bars[0].ts_opened 
        self.ts_closed = self.twist_bars[-1].ts_closed 
        self.index = 0 

    def power(self):
        """
        分型力度值，数值越大表示分型力度越大
        根据第三根K线与前两根K线的位置关系决定
        """
        power = 0
        first_twistbar = self.twist_bars[0]
        second_twistbar = self.twist_bars[1]
        third_twistbar = self.twist_bars[2]
        if third_twistbar is None:
            return power
        if self.mark_type == Mark.DING:
            if third_twistbar.high < (second_twistbar.high - (second_twistbar.high - second_twistbar.low) / 2):
                # 第三个K线的高点，低于第二根的50%以下
                power += 1
            if third_twistbar.low < first_twistbar.low and third_twistbar.low < second_twistbar.low:
                # 第三个最低点是三根中最低的
                power += 1
        elif self.mark_type == Mark.DI:
            if third_twistbar.low > (second_twistbar.low + (second_twistbar.high - second_twistbar.low) / 2):
                # 第三个K线的低点，低于第二根的50%之上
                power += 1
            if third_twistbar.high > first_twistbar.high and third_twistbar.high > second_twistbar.high:
                # 第三个最低点是三根中最低的
                power += 1
        return power

    def high(self):
        return max([item.high for item in self.twist_bars])
    
    def low(self):
        return min([item.low for item in self.twist_bars])


class LINE:
    def __init__(
        self,
        start,
        end,
        direction_type, ##Direction 
        power,
        is_confirm,
    ):
        self.start = start 
        self.end = end 
        self.direction_type = direction_type 
        self.power = power 
        self.is_confirm = is_confirm 

        self.ts_opened = self.start.ts_opened 
        self.ts_closed = self.end.ts_closed 


    def ding_high(self):
        return self.end.value if self.direction_type == Direction.UP else self.start.value

    def di_low(self):
        return self.end.value if self.direction_type == Direction.DOWN else self.start.value

    def dd_high_low(self):
        """
        返回线 顶底端点 的高低点
        """
        if self.direction_type == Direction.UP:
            return {
                'high': self.end.value, 
                'low': self.start.value
                }
        else:
            return {
                'high': self.start.value, 
                'low': self.end.value
                }

    def real_high_low(self):
        """
        返回线 两端 实际的 高低点
        """
        return {
            'high': self.high,
             'low': self.low
             }
    def angle(self) -> float:
        """
        计算线段与坐标轴呈现的角度（正为上，负为下）
        """
        # 计算斜率
        # convert to minute 
        duration = self.start.index - self.end.index 
        k = (self.start.value - self.end.value) / duration 
        # 斜率转弧度
        k = math.atan(k)
        # 弧度转角度
        j = math.degrees(k)
        return j


class ZS:
    def __init__(
        self,
        zs_type,
        zs_direction_type,
        start: FX, 
        end: FX = None, 
        zg: float = None, 
        zd: float = None,
        gg: float = None, 
        dd: float = None,
        level: int = 0, 
        is_high_kz: bool = False, 
        max_power: dict = None
    ):  
        self.zs_type = zs_type 
        self.zs_direction_type = zs_direction_type 
        self.start = start 
        self.end = end 
        self.zg = zg 
        self.zd = zd 
        self.gg = gg 
        self.dd = dd 
        self.level = level 
        self.is_high_kz = is_high_kz 
        self.max_power = max_power 
        self.lines =[] 
        self.line_num = len(self.lines) -1 
        self.line_num = len(self.lines)
        self.ts_opened = self.start.ts_opened 
        self.ts_closed = self.end.ts_closed 
        self.is_confirm = False 
        self.real = True 

    def add_line(
        self, 
        line: LINE
        ) -> bool:
        """
        添加 笔 or 线段
        """
        self.lines.append(line)
        return True

    def zf(self) -> float:
        """
        中枢振幅
        中枢重叠区间占整个中枢区间的百分比，越大说明中枢重叠区域外的波动越小
        """
        zgzd = self.zg - self.zd
        if zgzd == 0:
            zgzd = 1
        return (zgzd / (self.gg - self.dd)) * 100


class TradePoint:
    """
    买卖点对象
    """

    def __init__(
        self, 
        name: str, 
        zs: ZS
    ):
        self.name: str = name  # 买卖点名称
        self.zs: ZS = zs  # 买卖点对应的中枢对象

        self.ts_opened = self.zs.ts_opened 
        self.ts_closed = self.zs.ts_closed 

    def __str__(self):
        return 'TradePoint: %s ZS: %s' % (self.name, self.zs)


class BC:
    """
    背驰对象
    """

    def __init__(
        self, 
        _type: str, 
        zs: Union[ZS, None], 
        compare_line: LINE,
        compare_lines: List[LINE],
        bc: bool
    ):
        self.type: str = _type  # 背驰类型 （bi 笔背驰 xd 线段背驰 zsd 走势段背驰 pz 盘整背驰 qs 趋势背驰）
        self.zs: Union[ZS, None] = zs  # 背驰对应的中枢
        self.compare_line: LINE = compare_line  # 比较的笔 or 线段， 在 笔背驰、线段背驰、盘整背驰有用
        self.compare_lines: List[LINE] = compare_lines  # 在趋势背驰的时候使用
        self.bc = bc  # 是否背驰

    def __str__(self):
        return f'BC type: {self.type} bc: {self.bc} zs: {self.zs}'

class BI(LINE):
    """
    笔对象
    """

    def __init__(
        self, 
        start: FX, 
        end: FX = None, 
        direction_type: str = None,
        power: dict = None, 
        is_confirm: bool = None, 
        pause: bool = False, 
        default_zs_type: str = None,
    ):
        super().__init__(start, end, direction_type, power, is_confirm)
        self.pause = pause 
        self.default_zs_type = default_zs_type 
        self.trade_points: List[TradePoint] = []  # 买卖点
        self.bcs: List[BC] = []  # 背驰信息
        self.pause: bool = pause  # 笔是否停顿
        # 记录不同中枢下的背驰和买卖点
        self.zs_type_trade_points: Dict[str, List[TradePoint]] = {}
        self.zs_type_bcs: Dict[str, List[BC]] = {}

    def get_trade_points(self, zs_type: str = None) -> List[TradePoint]:
        if zs_type is None:
            return self.trade_points
        if zs_type not in self.zs_type_trade_points.keys():
            return []
        return self.zs_type_trade_points[zs_type]

    def get_bcs(self, zs_type: str = None) -> List[BC]:
        if zs_type is None:
            return self.bcs
        if zs_type not in self.zs_type_bcs.keys():
            return []
        return self.zs_type_bcs[zs_type]

    def add_trade_point(self, name: str, zs: ZS, zs_type: str) -> bool:
        """
        添加买卖点
        """
        trade_point_obj = TradePoint(name, zs)
        if zs_type == self.default_zs_type:
            self.trade_points.append(trade_point_obj)

        if zs_type not in self.zs_type_trade_points.keys():
            self.zs_type_trade_points[zs_type] = []
        self.zs_type_trade_points[zs_type].append(trade_point_obj)
        return True

    def add_bc(
            self,
            _type: str,
            zs: Union[ZS, None],
            compare_line: Union[LINE, None],
            compare_lines: List[LINE],
            bc: bool,
            zs_type: str
    ) -> bool:
        """
        添加背驰点
        """
        bc_obj = BC(_type, zs, compare_line, compare_lines, bc)
        if zs_type == self.default_zs_type:
            self.bcs.append(bc_obj)
        if zs_type not in self.zs_type_bcs.keys():
            self.zs_type_bcs[zs_type] = []
        self.zs_type_bcs[zs_type].append(bc_obj)

        return True

    def line_trade_points(self, zs_type: Union[str, None] = None) -> list:
        """
        返回当前线所有买卖点名称
        zs_type 如果等于  | ，获取当前笔所有中枢的买卖点 合集
        zs_type 如果等于  & ，获取当前笔所有中枢的买卖点 交集
        """
        if zs_type is None:
            return [m.name for m in self.trade_points]

        if zs_type == '|':
            trade_points = []
            for zs_type in self.zs_type_trade_points.keys():
                trade_points += self.line_trade_points(zs_type)
            return list(set(trade_points))
        if zs_type == '&':
            trade_points = self.line_trade_points()
            for zs_type in self.zs_type_trade_points.keys():
                trade_points = set(trade_points) & set(self.line_trade_points(zs_type))
            return list(trade_points)

        if zs_type not in self.zs_type_trade_points.keys():
            return []
        return [m.name for m in self.zs_type_trade_points[zs_type]]

    def line_bcs(self, zs_type: Union[str, None] = None) -> list:
        """
        返回当前线所有的背驰类型
        zs_type 如果等于  | ，获取当前笔所有中枢的买卖点 合集
        zs_type 如果等于  & ，获取当前笔所有中枢的买卖点 交集
        """
        if zs_type is None:
            return [_bc.type for _bc in self.bcs if _bc.bc]

        if zs_type == '|':
            bcs = []
            for zs_type in self.zs_type_bcs.keys():
                bcs += self.line_bcs(zs_type)
            return list(set(bcs))
        if zs_type == '&':
            bcs = self.line_bcs()
            for zs_type in self.zs_type_bcs.keys():
                bcs = set(bcs) & set(self.line_bcs(zs_type))
            return list(bcs)

        if zs_type not in self.zs_type_bcs.keys():
            return []
        return [_bc.type for _bc in self.zs_type_bcs[zs_type] if _bc.bc]

    def trade_point_exists(self, check_trade_points: list, zs_type: Union[str, None] = None) -> bool:
        """
        检查当前笔是否包含指定的买卖点的一个
        """
        trade_points = self.line_trade_points(zs_type)
        return len(set(check_trade_points) & set(trade_points)) > 0

    def bc_exists(self, bc_types: list, zs_type: Union[str, None] = None) -> bool:
        """
        检查是否有背驰的情况
        """
        bcs = self.line_bcs(zs_type)
        return len(set(bc_types) & set(bcs)) > 0


class TZXL:
    """
    feature sequence 
    """

    def __init__(
        self, 
        line: Union[LINE, None], 
        pre_line: LINE, 
        mark_type: Mark,
        high: float, 
        low: float, 
        line_broken: bool = False,
        is_confirm: bool
    ):
        self.line: Union[LINE, None] = line
        self.high: float = high
        self.low: float = low 
        self.pre_line: LINE = pre_line
        self.line_broken: bool = line_broken
        self.is_up_line: bool = False
        self.lines: List[LINE] = [line]
        self.is_confirm: bool = is_confirm

class XLFX:
    """
    序列分型,3笔成一个分型 
    """

    def __init__(
        self,
        mark_type: str, 
        high: float, 
        low: float, 
        line: LINE,
        jump: bool = False, 
        line_broken: bool = False,
        fx_high: float = None, 
        fx_low: float = None, 
        is_confirm: bool = True
    ):
        self.mark_type = mark_type
        self.high = high
        self.low = low
        self.line = line

        self.jump = jump  # 分型是否有缺口
        self.line_broken = line_broken  # 标记是否线破坏
        self.fx_high = fx_high  # 三个分型特征序列的最高点
        self.fx_low = fx_low  # 三个分型特征序列的最低点
        self.is_confirm = is_confirm  # 序列分型是否完成


class XD(LINE):
    """
    线段对象
    """

    def __init__(
        self, 
        start: FX, 
        end: FX, 
        start_line: LINE, 
        end_line: LINE = None, 
        direction_type: str = None,
        high: float = None,
        low: float = None,
        ding_fx: XLFX = None, 
        di_fx: XLFX = None, 
        power: dict = None,
        is_confirm: bool = True,
        default_zs_type: str = None
    ):
        super().__init__(start, end, high, low, direction_type, power, is_confirm)

        self.start_line: LINE = start_line  # 线段起始笔
        self.end_line: LINE = end_line  # 线段结束笔
        self.ding_fx: XLFX = ding_fx
        self.di_fx: XLFX = di_fx

        self.trade_points: List[TradePoint] = []  # 买卖点
        self.bcs: List[BC] = []  # 背驰信息

    def is_jump(self):
        """
        成线段的分型是否有缺口
        """
        if self.direction_type == Direction.UP:
            return self.ding_fx.jump
        else:
            return self.di_fx.jump

    def is_line_broken(self):
        """
        成线段的分数，是否背笔破坏（被笔破坏不等于线段结束，但是有大概率是结束了）
        """
        if self.direction_type == Direction.UP:
            return self.ding_fx.line_broken
        else:
            return self.di_fx.line_broken

    def get_trade_points(self, zs_type: str = None) -> List[TradePoint]:
        if zs_type is None:
            return self.trade_points
        if zs_type not in self.zs_type_trade_points.keys():
            return []
        return self.zs_type_trade_points[zs_type]

    def get_bcs(self, zs_type: str = None) -> List[BC]:
        if zs_type is None:
            return self.bcs
        if zs_type not in self.zs_type_bcs.keys():
            return []
        return self.zs_type_bcs[zs_type]

    def add_trade_point(self, name: str, zs: ZS, zs_type: str) -> bool:
        """
        添加买卖点
        """
        trade_point_obj = TradePoint(name, zs)
        if zs_type == self.default_zs_type:
            self.trade_points.append(trade_point_obj)
        if zs_type not in self.zs_type_trade_points.keys():
            self.zs_type_trade_points[zs_type] = []
        self.zs_type_trade_points[zs_type].append(trade_point_obj)
        return True

    def add_bc(
            self, _type: str, zs: Union[ZS, None],
            compare_line: LINE, compare_lines: List[LINE], bc: bool,
            zs_type: str
    ) -> bool:
        """
        添加背驰点
        """
        bc_obj = BC(_type, zs, compare_line, compare_lines, bc)
        if zs_type == self.default_zs_type:
            self.bcs.append(bc_obj)
        if zs_type not in self.zs_type_bcs.keys():
            self.zs_type_bcs[zs_type] = []
        self.zs_type_bcs[zs_type].append(bc_obj)
        return True

    def line_trade_points(self, zs_type: Union[str, None] = None) -> list:
        """
        返回当前线所有买卖点名称
        zs_type 如果等于  | ，获取当前笔所有中枢的买卖点 合集
        zs_type 如果等于  & ，获取当前笔所有中枢的买卖点 交集
        """
        if zs_type is None:
            return [m.name for m in self.trade_points]

        if zs_type == '|':
            trade_points = []
            for zs_type in self.zs_type_trade_points.keys():
                trade_points += self.line_trade_points(zs_type)
            return list(set(trade_points))
        if zs_type == '&':
            trade_points = self.line_trade_points()
            for zs_type in self.zs_type_trade_points.keys():
                trade_points = set(trade_points) & set(self.line_trade_points(zs_type))
            return list(trade_points)

        if zs_type not in self.zs_type_trade_points.keys():
            return []
        return [m.name for m in self.zs_type_trade_points[zs_type]]

    def line_bcs(self, zs_type: Union[str, None] = None) -> list:
        """
        返回当前线所有的背驰类型
        zs_type 如果等于  | ，获取当前笔所有中枢的买卖点 合集
        zs_type 如果等于  & ，获取当前笔所有中枢的买卖点 交集
        """
        if zs_type is None:
            return [_bc.type for _bc in self.bcs if _bc.bc]

        if zs_type == '|':
            bcs = []
            for zs_type in self.zs_type_bcs.keys():
                bcs += self.line_bcs(zs_type)
            return list(set(bcs))
        if zs_type == '&':
            bcs = self.line_bcs()
            for zs_type in self.zs_type_bcs.keys():
                bcs = set(bcs) & set(self.line_bcs(zs_type))
            return list(bcs)

        if zs_type not in self.zs_type_bcs.keys():
            return []
        return [_bc.type for _bc in self.zs_type_bcs[zs_type] if _bc.bc]

    def trade_point_exists(self, check_trade_points: list, zs_type: Union[str, None] = None) -> bool:
        """
        检查当前笔是否包含指定的买卖点的一个
        """
        trade_points = self.line_trade_points(zs_type)
        return len(set(check_trade_points) & set(trade_points)) > 0

    def bc_exists(self, bc_types: list, zs_type: Union[str, None] = None) -> bool:
        """
        检查是否有背驰的情况
        """
        bcs = self.line_bcs(zs_type)
        return len(set(bc_types) & set(bcs)) > 0

    def is_confirm(self) -> bool:
        """
        线段是否完成
        """
        return self.ding_fx.is_confirm if self.direction_type == Direction.UP  else self.di_fx.is_confirm
