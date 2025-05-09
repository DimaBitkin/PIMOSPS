from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple
import tkinter as tk
import re
from tkinter import messagebox

# -------------------- Модель процессора --------------------
@dataclass
class CPU:
    # 16 регистров R0–R15, инициализируются нулями
    registers: Dict[str, int] = field(default_factory=lambda: {f"R{i}": 0 for i in range(16)})
    # Простая память в виде словаря (адрес: значение)
    memory: Dict[int, int] = field(default_factory=dict)
    pc: int = 0      # Program Counter
    lr: int = 0      # Link Register (используется при BL — переход с возвратом)
    # Флаги состояния: N — отрицательный, Z — ноль, C — перенос, V — переполнение
    flags: Dict[str, int] = field(default_factory=lambda: {'N': 0, 'Z': 0, 'C': 0, 'V': 0})
    # Конвейер команд: 5 стадий — Fetch, Decode, Execute, Memory, WriteBack
    pipeline: List[str] = field(default_factory=lambda: ["" for _ in range(5)])
    pending_write: Optional[Tuple[str, int]] = None  # Хранит (регистр, значение) до WriteBack
    #флаг перехода 
    branch_taken: bool = False

    def reset(self):
        # Сброс процессора: заново инициализировать все поля
        self.__init__()

# -------------------- Парсер инструкций --------------------
def parse_instruction(line: str) -> List[str]:
    return line.strip().replace(',', '').split()

# -------------------- Выполнение команд --------------------
def execute(cpu: CPU, instr: List[str]):
    if not instr:
        return

    op = instr[0].upper()

    def get_val(x):
        if x.startswith('R'):
            return cpu.registers.get(x, 0)
        return int(x.strip('#'))

    def get_imm(x):
        return int(x.strip('#'))

    def check_condition(cond):
        Z, N = cpu.flags['Z'], cpu.flags['N']
        if cond == 'EQ':
            return Z == 1
        elif cond == 'NE':
            return Z == 0
        elif cond == 'GT':
            return Z == 0 and N == 0
        elif cond == 'LT':
            return N == 1
        elif cond == 'GE':
            return N == 0
        elif cond == 'LE':
            return Z == 1 or N == 1
        else:
            return False

    try:
        if op == 'MOV':
            cpu.pending_write = (instr[1], get_val(instr[2]))

        elif op == 'ADD':
            cpu.pending_write = (instr[1], get_val(instr[2]) + get_val(instr[3]))

        elif op == 'SUB':
            cpu.pending_write = (instr[1], get_val(instr[2]) - get_val(instr[3]))

        elif op == 'CMP':
            result = get_val(instr[1]) - get_val(instr[2])
            cpu.flags['Z'] = int(result == 0)
            cpu.flags['N'] = int(result < 0)

        elif op == 'LDR':
            addr = get_val(instr[2])
            cpu.pending_write = (instr[1], cpu.memory.get(addr, 0))

        elif op == 'STR':
            addr = get_val(instr[2])
            cpu.memory[addr] = get_val(instr[1])

        elif op in ['BEQ', 'BNE', 'BGT', 'BLT', 'BGE', 'BLE']:
            cond = op[1:]
            if check_condition(cond):
                cpu.pc = int(instr[1])
                cpu.branch_taken = True

        elif op == 'B':
            cpu.pc = int(instr[1])
            cpu.branch_taken = True

        elif op == 'BL':
            cpu.lr = cpu.pc
            cpu.pc = int(instr[1]) 
            cpu.branch_taken = True

        elif op == 'BX':
            if instr[1].upper() == 'LR':
                cpu.pc = cpu.lr 
            else:
                cpu.pc = get_val(instr[1]) 
                cpu.branch_taken = True

        elif op == 'NOP':
            pass

        else:
            raise ValueError(f"Unknown instruction: {op}")

    except IndexError:
        raise ValueError(f"Недостаточно аргументов в инструкции: {' '.join(instr)}")
    except Exception as e:
        raise ValueError(f"Ошибка при выполнении инструкции {' '.join(instr)}: {str(e)}")


# -------------------- UI-интерфейс --------------------
class SimulatorApp:
    def __init__(self, root):
        self.root = root
        self.cpu = CPU()
        self.instructions: List[List[str]] = []

        root.title("ARM Instruction Simulator")

        self.instr_input = tk.Text(root, height=10, width=50)
        self.instr_input.pack()
        # После каждого нажатия клавиши вызывается метод syntax_highlight — он будет подсвечивать ключевые слова и ошибки в тексте.
        self.instr_input.bind("<KeyRelease>", self.syntax_highlight)

        # Определяем стили для подсветки
        self.instr_input.tag_configure("keyword", foreground="blue")
        self.instr_input.tag_configure("register", foreground="dark green")
        self.instr_input.tag_configure("error", background="red", foreground="white")

        #Создаем рамку (LabelFrame) с заголовком — здесь будут вводиться начальные значения регистров.
        self.reg_frame = tk.LabelFrame(root, text="Начальные значения регистров")
        self.reg_frame.pack()
        self.reg_entries = {}
        for i in range(16):
            reg = f"R{i}"
            lbl = tk.Label(self.reg_frame, text=reg)
            lbl.grid(row=i // 8, column=(i % 8) * 2)
            ent = tk.Entry(self.reg_frame, width=5)
            ent.grid(row=i // 8, column=(i % 8) * 2 + 1)
            self.reg_entries[reg] = ent

        self.load_button = tk.Button(root, text="Загрузить", command=self.load_instructions)
        self.load_button.pack()

        self.step_button = tk.Button(root, text="Следующий такт", command=self.step)
        self.step_button.pack()

        self.reset_button = tk.Button(root, text="Сброс", command=self.reset)
        self.reset_button.pack()

        self.state_display = tk.Text(root, height=30, width=70)
        self.state_display.pack()
        self.update_display()

    def syntax_highlight(self, event=None):
        text = self.instr_input.get("1.0", tk.END)
        self.instr_input.tag_remove("keyword", "1.0", tk.END)
        self.instr_input.tag_remove("register", "1.0", tk.END)
        self.instr_input.tag_remove("error", "1.0", tk.END)

        lines = text.strip().split("\n")
        keywords = {'MOV', 'ADD', 'SUB', 'CMP', 'B', 'BL', 'BX', 'LDR', 'STR', 'NOP'}
        #Регулярное выражение для поиска имен регистров R0 – R15.\b — граница слова, ([0-9]|1[0-5]) — цифра от 0 до 9 или от 10 до 15.
        reg_pattern = r'\bR([0-9]|1[0-5])\b'
        
        for lineno, line in enumerate(lines):
            line_start = f"{lineno+1}.0"
            line_end = f"{lineno+1}.end"

            # Проверка синтаксиса
            tokens = parse_instruction(line)
            if tokens and tokens[0].upper() not in keywords:
                self.instr_input.tag_add("error", line_start, line_end)

            # Подсветка ключевых слов
            for kw in keywords:
                for match in re.finditer(r'\b' + kw + r'\b', line, re.IGNORECASE):
                    start = f"{lineno+1}.{match.start()}"
                    end = f"{lineno+1}.{match.end()}"
                    self.instr_input.tag_add("keyword", start, end)

            # Подсветка регистров
            for match in re.finditer(reg_pattern, line, re.IGNORECASE):
                start = f"{lineno+1}.{match.start()}"
                end = f"{lineno+1}.{match.end()}"
                self.instr_input.tag_add("register", start, end)
    #загружает инструкции из текстового поля и начальные значения регистров в модель процессора.
    def load_instructions(self):
        lines = self.instr_input.get("1.0", tk.END).strip().split('\n')
        self.instructions = [parse_instruction(line) for line in lines if line.strip()]
        self.cpu.pc = 0
        self.cpu.pipeline = ["" for _ in range(5)]

        # Применение значений регистров из формы
        for reg, entry in self.reg_entries.items():
            val = entry.get()
            if val:
                try:
                    self.cpu.registers[reg] = int(val)
                except ValueError:
                    messagebox.showerror("Ошибка", f"Неверное значение для {reg}: {val}")
                    return

        self.update_display()
    #реализует один такт работы конвейера ARM-процессора:
    def step(self):
        # Обработка конвейера
        self.cpu.pipeline.pop()
        if self.cpu.pc < len(self.instructions):
            instr_str = " ".join(self.instructions[self.cpu.pc])
        else:
            instr_str = ""
        
        #Новая инструкция (или пустая строка) помещается в начало конвейера — стадию Fetch.
        self.cpu.pipeline.insert(0, instr_str)

        # Исполнение только когда команда на стадии Execute
        if self.cpu.pipeline[2]:
            try:
                execute(self.cpu, parse_instruction(self.cpu.pipeline[2]))
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return
        # WriteBack stage (index 4)
        if self.cpu.pending_write:
            reg, value = self.cpu.pending_write
            self.cpu.registers[reg] = value
            self.cpu.pending_write = None

        if not self.cpu.branch_taken and self.cpu.pc < len(self.instructions):
            self.cpu.pc += 1

        self.cpu.branch_taken = False  # сброс флага

        self.update_display()

    #Метод вызывается при нажатии на кнопку «Сброс». Он полностью очищает состояние эмулятора.
    def reset(self):
        self.cpu.reset()
        self.instructions = []
        #self.instr_input.delete("1.0", tk.END)
        for entry in self.reg_entries.values():
            entry.delete(0, tk.END)
        self.update_display()

    def update_display(self):
        self.state_display.delete("1.0", tk.END)
        self.state_display.insert(tk.END, "== Регистры ==\n")
        for i in range(16):
            reg = f"R{i}"
            self.state_display.insert(tk.END, f"{reg}: {self.cpu.registers[reg]}\n")
        self.state_display.insert(tk.END, f"PC: {self.cpu.pc}\n")
        self.state_display.insert(tk.END, f"LR: {self.cpu.lr}\n")
        self.state_display.insert(tk.END, "\n== Флаги ==\n")
        for flag, val in self.cpu.flags.items():
            self.state_display.insert(tk.END, f"{flag}: {val}  ")

        self.state_display.insert(tk.END, "\n\n== Конвейер ==\n")
        stages = ["Fetch", "Decode", "Execute", "Memory", "WriteBack"]
        for i, instr in enumerate(self.cpu.pipeline):
            self.state_display.insert(tk.END, f"{stages[i]}: {instr}\n")

        self.state_display.insert(tk.END, "\n== Память ==\n")
        for addr in sorted(self.cpu.memory.keys()):
            self.state_display.insert(tk.END, f"[{addr}]: {self.cpu.memory[addr]}\n")

# -------------------- Запуск --------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = SimulatorApp(root)
    root.mainloop()
