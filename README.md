## Отчет

* Кривошеев Андрей Александрович, P33111
* `lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob5 | 8bit`
* Без усложнения


#### Пояснение варианта

---

* `lisp` -- Язык программирования. Синтаксис -- синтаксис языка Lisp. S-exp.
  * требуется поддержка функций и/или процедур;
  * необходимо объяснить, как происходит отображение сложных выражений на регистры и память;
  * необходимо продемонстрировать работу транслятора в случае, если количества регистров недостаточно для реализации алгоритма.
  * Любое выражение (statement) -- expression. Примеры корректного кода (с точностью до ключевых слов):
    * (print (if (= 1 x) "T" "F"))
    * (setq x (if (= 1 x) 2 3))
    * (setq x (loop ...))
    * (print (seq x 13))
  * Необходимо объяснить и продемонстрировать, что любое выражение (statement) -- expression.

* `acc` -- Архитектура -- система команд должна быть выстроена вокруг аккумулятора.
  * Инструкции -- изменяют значение, хранимое в аккумуляторе.
  * Ввод-вывод осуществляется через аккумулятор.

* `harv` -- Архитектура организации памяти -- Гарвардская архитектура.

* `hw` -- Control Unit -- hardwired. Реализуется как часть модели.

* `instr` -- Точность модели -- процессор необходимо моделировать с точностью до каждой инструкции (наблюдается состояние после каждой инструкции).

* `binary` -- Представление машинного кода -- бинарное представление.
  * Требуются настоящие бинарные файлы, а не текстовые файлы с 0 и 1.
  * Требуется отладочный вывод в текстовый файл вида:
    ```text
    <address> - <HEXCODE> - <mnemonica> 
    20 - 03340301 - add #01 <- 34 + #03
    ```
    
* `stream` -- Ввод-вывод
  * Ввод-вывод осуществляется как поток токенов. Есть в примере. Логика работы:
    * при старте модели у вас есть буфер, в котором представлены все данные ввода (['h', 'e', 'l', 'l', 'o']);
    * при обращении к вводу (выполнение инструкции) модель процессора получает "токен" (символ) информации;
    * если данные в буфере кончились -- останавливайте моделирование;
    * вывод данных реализуется аналогично, по выполнении команд в буфер вывода добавляется ещё один символ;
    * по окончании моделирования показать все выведенные данные;
    * логика работы с буфером реализуется в рамках модели на Python.

* `mem` -- Ввод-вывод ISA -- memory-mapped (порты ввода-вывода отображаются в память и доступ к ним осуществляется штатными командами),
  * отображение портов ввода-вывода в память должно конфигурироваться (можно hardcode-ом).

* `cstr` -- Поддержка строк -- Null-terminated (C string)

* `prob5` -- Алгоритм -- Smallest multiple. [Project Euler. Problem 5](https://projecteuler.net/problem=5)

* `8bit` -- машинное слово -- 8 бит (как для памяти команд, так и для памяти данных, если они разделены).

---


### Язык программирования

Код состоит из последовательности выражений, заключенных в скобки (`(st ...)`). Выражение состоит из названия и агрументов, 
следующих последовательно друг за другом через пробел (`(name <arg1> <arg2>)`). В качестве аргументов могут выступать также выражения (`(func (add ...) ...)`),
переменные (`(submit variable1 variable2 ...)`) или числовые (`123`) или строковые литералы (`"str"`). Пробельные символы и переносы строк игнорируются.

Любое выражение/переменная/литерал имеют возвращаемое значение (кроме объявления функции).

Объявление функции выглядит, как обычное выражение, включающее название-аргументы-тело, также включает в себя объявление типов аргументов и
тип возвращаемого значения (`(defun func:i (arg1:i arg2:s) (...))`). Тело функция - ровно одно выражение.

Типов данных 2: 64-битное знаковое число (`i`), строка (`s`). Соответственно, каждое выражение возвращает значение одного из двух типов.

На уровне языка поддерживаются математические выражения "+" - сложение, "-" - вычитание, "*" - умножение, 
"/" - целочисленное деление, "mod" - остаток от деление (`(+ 1 2 (/ 5 3))`). Все аргументы математических операций имеют тип `i`.

Поддерживаются операторы сравнения "=" - равенство, ">=" - больше или равно. Все аргументы имеют тип `i`. 
Возвращаемое значение 1 - истина, 0 - ложь.

Переменные можно определить при помощи функции "set" (`(set <label> val:i|s)`). Функция возвращает установленное
значение. Область видимости данных - глобальная.

Также поддерживается условная конструкция "if" (`(if cond:i <opt1> <opt2>)`). `opt1` и `opt2` должны иметь одинаковый
возвращаемый тип - он и является возвращаемым типом всей конструкции. Истина или ложь определяется по значению `cond`,
равно 0 - будет выполнен код `opt2`, иначе `opt1`.

Определены функции для работы с вводом выводом: `(printi arg:i)`, `(prints arg:i)`, `(read)`. Функции вывода 
возвращают количество выведенных символов, а функция чтения - прочитанную строку.

Код выполняется последовательно сверху вниз. Внутри выражений код выполняется *справа налево*.


### Организация памяти

Память разделена на память команд и память данных.

Память инструкций имеет примерно следующий вид:

```text

    +------------------------+
    | 0: br n                |
    | 1: <func1 code>        |
    | ...                    |
    | k: <func2 code>        |
    | ...                    |
    | n: <main code>         |
    | ...                    |
    | m: halt                |
    +------------------------+
    
```

Машинное слово - 32 бита (8 бит - код операции, 4 бит - тип адресации, 20 бит - значение для адресации).

Первая инструкция - переход к основной программе (которая был странслирована из исходного кода, написанного программистом).
Далее хранятся друг за другом используемые программой функции. После всех функции расположены инструкции основной программы.
Последняя инструкция - `halt`.

Память данных имеет примерно следующий вид:

```text

    +------------------------+
    | 0: const 1             |
    | 1: const 2             |
    | ...                    |
    | v+0: variable 1        |
    | v+1: variable 2        |
    | ...                    |
    | b+0: anon buffer 1     |
    | b+1: anon buffer 2     |
    | ...                    |
    | ...                    |
    | n: stack start         |
    +------------------------+   
    
```

Машинное слово - 64 бита.

Сначала располагаются константы (числовые - 1 слово, строковые - количество символов + 1 слов). Далее находятся
переменные (по 1 машинному слову). Далее выделены анонимные буферы для некоторых команд, которые просто представляют
несколько размапленных друг за другом слов (например, `read` выделяет буфер для чтения данных из ввода). В конце 
памяти данных находится место для стека, он "растет" от конца памяти к началу.

Программист с регистрами и памятью напрямую не работает.

Строковые литералы всегда считаются константами и записываются в область констант. Числовые литералы считаются
константой, если не влазят в 20 бит (то есть больше 2^19 - 1 или меньше -2^19), будут помещены в область констант.
Остальные числа будут загружены в регистры при помощи непосредственной адресации.

Порядок расположения констант, переменных определяется по алфавитному порядку (числа переводятся в строку как есть).
Порядок расположения буферов определяется порядком встречи в коде.

Во время трансляции исходной программы если заявляется необходимость выделения константы/переменной/буфера, то в
машинном коде временно записывается символ (идентификатор) выделенной памяти. На конечном этапе трансляции происходит 
маппинг всех запрошенных констант/переменных/буферов на память, а все идентификаторы в машинном коде заменяются на
непосредственный адрес.

В стек ничего не отображается.

Поскольку архитектура процессора - аккумуляторная, то все переменные отображаются в память, а при необходимости
обращения в аккумулятор загружается их значение. При вызове функций первый аргумент (если есть) передается через аккумулятор,
а последующие (если есть) передаются через стек.


### Система команд

Типов данных 2: 64-битное знаковое число, строка. Внутри процессора число представляет из себя свое значение, а строка -
указатель по абсолютному адресу на первый символ.

Регистры:

| Название | Разрядность | Описание                                                                    |
|----------|-------------|-----------------------------------------------------------------------------|
| acr      | 64          | аккумулятор                                                                 |
| ipr      | 64          | указатель на адрес исполняемой инструкции                                   |
| inr      | 32          | содержит текущую исполняемую инструкцию                                     |
| bur      | 64          | буферный регистр, используется для проведения операций между arc, ipr, inr  |
| adr      | 64          | регистр адреса, используется для чтения/записи                              |
| dar      | 64          | регистр данных, используется для чтения/записи                              |
| spr      | 64          | указатель на вершину стека                                                  |
| flr      | 4           | регистр флагов, содержит флаги по результатам операции (NZVC)               |

Доступ к памяти данных осуществляется при помощи регистров adr, dar. Для чтения необходимо задать адрес ячейки для чтения.
Для записи необходимо задать адрес и значение для записи.

Ввод/вывод замаплен на память. Для вывода необходимо выполнить запись в ячейку памяти по определенному адресу. Для ввода
необходимо выполнить чтение определенной (другой) ячейки памяти.

Имеется 5 типов адресации:
  * непосредственная
  * абсолютная
  * относительная (ipr)
  * относительная (spr)
  * относительная косвенная (spr)

Инструкции делятся на 3 типа: без аргумента, адресные, по значению.
Инструкции без аргумента не требуют адресации. Адресные используют аргумент адреса с которым будет выполнено действие.
Инструкции по значению используют адресацию, как адрес значения, с которым будет выполнено действие.

Набор инструкций:

| Инструкция  | Описание                                                                  |
|-------------|---------------------------------------------------------------------------|
| noop        | ничего не выполняется                                                     |
| halt        | останов                                                                   |
| ld <val>    | загрузка значения из памяти в аккумулятор                                 |
| st <addr>   | запись значения аккумулятора в память                                     |
| call <addr> | вызвать функции по адресу (адрес возврата положить на стек)               |
| ret         | возвращение из функции (адрес возврата снять со стека)                    |
| push        | положить значение аккумулятора на стек                                    |
| pop         | снять значение со стека и записать в аккумулятор                          |
| popn        | снять значение со стека без записи (spr--)                                |
| cmp <val>   | установить флаги по результату операции вычитания аккумулятора и значения |
| brz <addr>  | переход по адресу если установлен флаг Z (zero)                           |
| brge <addr> | переход по адресу если значение больше или равно (S == O)                 |
| br <addr>   | безусловный переход по адресу                                             |
| inc <addr>  | инкремент значения по адресу (без затрагивания аккумулятора)              |
| dec <addr>  | декремент значения по адресу (без затрагивания аккумулятора)              |
| mod <val>   | деление по модулю аккумулятора и значения                                 |
| add <val>   | прибавить к аккумулятор значения                                          |
| sub <val>   | вычесть из аккумулятора значение                                          |
| mul <val>   | умножить аккумулятор на значение                                          |
| div <val>   | целочисленно поделить аккумулятор на значение                             |
| inv         | инвертировать значение аккумулятора                                       |


Поток управления:
  * вызов `call` или возврат `ret` из функции
  * условные `brz`, `brge` и безусловные `br` переходы
  * инкремент `ipr` после любой другой инструкции

Машинный код сериализуется в список 32 битных слов. Одно слово - одна инструкция. Адресация инструкций с 0 по порядку.

Кодирование инструкций:
  * 8 бит - код операции
  * 4 бит - тип адресации (для команд с аргументом)
  * 20 бит - значение для адресации (для команд с аргументом)

Типы данных в модуле [isa](isa.py):
  * Opcode - перечисление кодов операций
  * AddressType - перечисление типов адресации
  * Address - структура объединение типа адресации и значения
  * Term - структура для описания одной инструкции


### Транслятор

Интерфейс командной строки: `translator.py <input_file> <target_file>` (файл с отладочным выводом - `<target_file>.debug`)

Реализовано в модуле: [translator](translator.py)

Этапы трансляции:
  * чтение исходного кода
  * трансформация текста в последовательность токенов с тегами
  * преобразование токенов в абстрактное дерево (выражение - вершина, литерал/идентификатор - лист)
  * вычленение Statement на основе дерева и валидация и проверка корректности типов
  * генерация машинного кода
  * запись полученного кода в бинарный файл 
  * запись полученного кода в текстовый файл

Правила генерации машинного кода:
  * каждый Statement должен к концу исполнения должен установить в аккумулятор возвращаемое значение, а также
    вернуть spr в то состояние, в котором он был в начале исполнения
  * при вызове функции вызывающая сторона обязана установить все требуемые аргументы, а также после возврата
    снять со стека все положенные аргументы
  * Statement содержащий переменную или литерал должен загрузить в аккумулятор нужное значение
  * Statement с несколькими агрументами для вычисления производят генерацию кода вычисления с последнего к первому
    (справа налево)
  * при необходимости выделения места в памяти под константу/литерал/буфер, происходит обращение в специальный
    класс, который запоминает все обращения и выдает идентификаторы для последующей генерации памяти данных
    и замены символов на непосредственные адреса
  * при трансляции кода для функции каждый Statement должен учитывать то, где сейчас находятся (в аккумуляторе, в стеке)
    аргументы и помещать их на стек в случае необходимости, а также отслеживать их актуальное положение 
    (смещение относительно spr)
  * после завершения генерации кода, все символы в инструкциях заменяются на адреса в зависимости от их маппинга
    в память


### Модель процессора

Интерфейс командной строки: `machine.py <machine_code_file> <input_file>`

Реализовано в модуле: [machine](machine.py)

#### Data path

```text

                                                                        +-------+<--- latch
                                                            +-----------|  bur  |
                                                            |           +-------+<-----------+
                                                            |                                |
                      latch --->+-------+                   |           +-------+<--- latch  |
                                |  acr  |-----------+       | +---------|  adr  |            |
                   +----------->+-------+           |       | |         +-------+<-----------+
                   |                                |       | |                              |
                   |  latch --->+-------+           |       | |         +-------+<--- latch  |
                   |            |  ipr  |--------+  |       | | +-------|  dar  |   +-----+  |
                   +----------->+-------+        |  |       | | |       +-------+<--| MUX |--+
                   |                             |  |       | | |           sel --->+-----+  |
                   |            +-------+        |  | 0   0 | | |       +-------+<--- latch  |
                   |            |  inr  |-----+  |  | |   | | | | +-----|  spr  |            |
                   |            +-------+     |  |  | |   | | | | |     +-------+<-----------+
                   |                          v  v  v v   v v v v v                          |
                   |                        +---------+   +---------+                        |
                   |                sel --->|   MUX   |   |   MUX   |<--- sel                |
                   |                        +---------+   +---------+                        |
                   |                             |             |                             |
                   |                             v             v                             |
         latch     |           inv_left --->+---------+   +---------+<--- inv_right          |
           |       |          extend_20 --->|          \_/          |<--- (+1)               |
           v       |                        |                       |                        |  
       +-------+   |             opcode --->|          ALU          |<--- inv_out            |
       |  flr  |<----------------------------\                     /                         |
       +-------+   |                          +-------------------+                          |
                   |                                    |                                    |
                   +------------------------------------+------------------------------------+

```

#### Взаимодействие с памятью данных

```text
            
                                 read     write
                                   |        |
                                   v        v
              +-------+          +------------+
              |  adr  |--------->|            |<------- input
              +-------+          |            |
                                 |    data    |-------> output
                                 |   memory   |
              +-------+          |            |
   latch ---->|  dar  |--------->|            |-------+
              +-------+          +------------+       |
                   ^                                  |
                   |   +-----+                        |
                   +---| MUX |------------------------+
                       +-----+
                          ^
                          |
                         sel
                         
```

Реализован в классе `DataPath`.

Сигналы:
  * `read_data_memory` - чтение данных по адресу `adr` в `dar`:
    * из памяти данных (`dar <- mem[adr]`)
    * из порта ввода `input`:
      * извлечь из входного буфера токен и подать его на выход
      * если буфер пуст - исключение
  * `write_data_memory` - запись данных `dar` по адресу `adr`:
    * в памяти данных (`mem[adr] <- dar`)
    * в порт вывода `output`:
      * записать значение `dar` в буфер вывода
  * `latch_acr`, `latch_ipr`, `latch_bur`, `latch_adr`, `latch_spr` - заgисать значение с выхода ALU в регистр
  * `sel_dar` - выбрать значение с выхода ALU или из памяти данных для записи в `dar`
  * `latch_dar` - записать выставленное значение в `dar`
  * `latch_flr` - записать значения флагов операции суммы из ALU в `flr`
  * `sel_left`, `sel_right` - выбрать значение для левого и прового входов ALU
  * `alu_opcode` - выбор операции для осуществления на ALU (sum, mul, div, mod)
  * `alu_inv_left`, `alu_inv_right`, `alu_inv_out` - инвертировать левый вход, правый вход, выход из ALU соответственно
  * `alu_extend_20` - расширить знак 20 бит левого входа ALU
  * `alu_plus_1` - прибавить 1 к операции суммы ALU


#### ControlUnit (взаимодействие с памятью инструкций)

```text

                                                                   +---------+
                                                               +-->|  step   |--+                 input   output
                                                   latch       |   | counter |  |                   |       ^
         latch             +--------------+          |         |   +---------+  v                   v       |
           |      read --->|              |          v         |     +---------------+  signals   +-----------+
           v               | instructuion |       +-------+    +-----|  instruction  |----------->| DataPath* |
        +-------+          |    memory    |------>|  inr  |--------->|    decoder    |            +-----------+
   +--->|  ipr  |--------->|              |       +-------+          +---------------+               |    |
   |    +-------+          +--------------+                                        ^                 |    |
   |                                                                               |                 |    |
   |                                                                               +-----------------+    |
   |                                                                                 feedback_signals     |
   |                                                                                                      |
   +------------------------------------------------------------------------------------------------------+
                                                                                            
```

Реализован в классе `ControlUnit`.

Особенности:
  * hardwired (реализован полностью на Python, для каждого типа инструкции можно однозначно 
    определить сигналы и их порядок для каждого такта исполнения инструкции)
  * step counter - необходим для много-тактовых инструкций

Сигналы:
  * `read_instruction_memory` - чтение данных по адресу `ipr` в `inr` (`inr <- mem[ipr]`)
  * `latch_inr` - записать значение из памяти инструкций в `inr`
  * signals - управляющие сигналы в DataPath
  * feedback_signals - сигналы обратной связи (`flr`)

Шаг декодирования и исполнения одной инструкции выполняется в функции `execute_next_instruction`.
Во время шага в журнал моделирования записывается исполняемая инструкция и состояние модели на конец исполнения.

Цикл симуляции осуществляется в функции simulation.

Остановка моделирования осуществляется при превышении лимита количества выполняемых инструкций, 
при отсутствии данных для чтения из порта ввода, при выполнении инструкции halt.

### Тестирование

Для тестирования выбраны 7 алгоритмов:
1. [hello](./golden/hello.yml) - [source](./examples/hello)
2. [echo](./golden/echo.yml) - [source](./examples/echo)
3. [hello_user_name](./golden/hello_user_name.yml) - [source](./examples/hello_user_name)
4. [prob5](./golden/prob5.yml) - [source](./examples/prob5)
5. [if](./golden/if.yml)
6. [math](./golden/math.yml)
7. [func](./golden/func.yml)

Golden-тесты реализованы в [integration_test.py](integration_test.py), конфигурация к ним 
находится в директории [golden](./golden).

CI:

```yaml
lint:
  stage: test
  image:
    name: ryukzak/python-tools
    entrypoint: [""]
  script:
    - poetry install
    - coverage run -m pytest --verbose
    - find . -type f -name "*.py" | xargs -t coverage report
    - ruff format --diff .
    - ruff check .
```

где:
* `ryukzak/python-tools` - docker образ содержит все необходимые для проверки утилиты
* `poetry` - управления зависимостями
* `coverage` - формирование отчёта об уровне покрытия исходного кода
* `pytest` - утилита для запуска тестов
* `ruff` - утилита для форматирования и проверки стиля кодирования

Пример использования и журнал работы процессора на примере `hello`:

```bash
% cat ./examples/hello
(prints "Hello, world!")
% cat ./examples/input/hello
% python3 translator.py ./examples/hello target     
LoC: 1 code byte: 186 code instr: 16
% cat target.debug 
##### Data section #####
<address> - <length> - <data>
0x000 - 14      - Hello, world!

##### Instruction section #####
<address> - <hexcode> - <mnemonica>
#:
0x000 - 0x0c20000d - br *0xd                    (start)
prints:
0x001 - 0x06000000 - push
0x002 - 0x06000000 - push
0x003 - 0x02500000 - ld **spr
0x004 - 0x09100000 - cmp #0x0
0x005 - 0x0a300004 - brz *ipr+0x4
0x006 - 0x032015b4 - st *0x15b4
0x007 - 0x0d400000 - inc *spr
0x008 - 0x0c3ffffb - br *ipr-0x5
0x009 - 0x07000000 - pop
0x00a - 0x11400000 - sub *spr
0x00b - 0x08000000 - popn
0x00c - 0x05000000 - ret
start:
0x00d - 0x02100000 - ld #0x0                    (Hello, world!)
0x00e - 0x04200001 - call *0x1                  (prints)
0x00f - 0x01000000 - halt
% python3 machine.py target ./examples/input/hello
DEBUG:root:br *0xd
DEBUG:root:tick=1 acr=0x0 ipr=0xd inr=0xc20000d bur=0x0 adr=0x0 dar=0x0 spr=0x1fff flr=0x0 stack_top=?
DEBUG:root:ld #0x0
DEBUG:root:tick=3 acr=0x0 ipr=0xe inr=0x2100000 bur=0x0 adr=0x0 dar=0x0 spr=0x1fff flr=0x0 stack_top=?
DEBUG:root:call *0x1
DEBUG:root:tick=7 acr=0x0 ipr=0x1 inr=0x4200001 bur=0x0 adr=0x1ffe dar=0xf spr=0x1ffe flr=0x0 stack_top=0xf
DEBUG:root:push
DEBUG:root:tick=10 acr=0x0 ipr=0x2 inr=0x6000000 bur=0x0 adr=0x1ffd dar=0x0 spr=0x1ffd flr=0x0 stack_top=0x0
DEBUG:root:push
DEBUG:root:tick=13 acr=0x0 ipr=0x3 inr=0x6000000 bur=0x0 adr=0x1ffc dar=0x0 spr=0x1ffc flr=0x0 stack_top=0x0
DEBUG:root:ld **spr
DEBUG:root:tick=18 acr=0x48 ipr=0x4 inr=0x2500000 bur=0x0 adr=0x0 dar=0x48 spr=0x1ffc flr=0x0 stack_top=0x0
DEBUG:root:cmp #0x0
DEBUG:root:tick=20 acr=0x48 ipr=0x5 inr=0x9100000 bur=0x0 adr=0x0 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=21 acr=0x48 ipr=0x6 inr=0xa300004 bur=0x0 adr=0x0 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:st *0x15b4
DEBUG:root:output: '' <- 'H'
DEBUG:root:tick=24 acr=0x48 ipr=0x7 inr=0x32015b4 bur=0x0 adr=0x15b4 dar=0x48 spr=0x1ffc flr=0x1 stack_top=0x0
DEBUG:root:inc *spr
DEBUG:root:tick=28 acr=0x48 ipr=0x8 inr=0xd400000 bur=0x0 adr=0x1ffc dar=0x1 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=30 acr=0x48 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x1 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:ld **spr
DEBUG:root:tick=35 acr=0x65 ipr=0x4 inr=0x2500000 bur=0x8 adr=0x1 dar=0x65 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:cmp #0x0
DEBUG:root:tick=37 acr=0x65 ipr=0x5 inr=0x9100000 bur=0x8 adr=0x1 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=38 acr=0x65 ipr=0x6 inr=0xa300004 bur=0x8 adr=0x1 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:st *0x15b4
DEBUG:root:output: 'H' <- 'e'
DEBUG:root:tick=41 acr=0x65 ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x65 spr=0x1ffc flr=0x1 stack_top=0x1
DEBUG:root:inc *spr
DEBUG:root:tick=45 acr=0x65 ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x2 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=47 acr=0x65 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x2 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:ld **spr
DEBUG:root:tick=52 acr=0x6c ipr=0x4 inr=0x2500000 bur=0x8 adr=0x2 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:cmp #0x0
DEBUG:root:tick=54 acr=0x6c ipr=0x5 inr=0x9100000 bur=0x8 adr=0x2 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=55 acr=0x6c ipr=0x6 inr=0xa300004 bur=0x8 adr=0x2 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:st *0x15b4
DEBUG:root:output: 'He' <- 'l'
DEBUG:root:tick=58 acr=0x6c ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0x2
DEBUG:root:inc *spr
DEBUG:root:tick=62 acr=0x6c ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x3 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=64 acr=0x6c ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x3 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:ld **spr
DEBUG:root:tick=69 acr=0x6c ipr=0x4 inr=0x2500000 bur=0x8 adr=0x3 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:cmp #0x0
DEBUG:root:tick=71 acr=0x6c ipr=0x5 inr=0x9100000 bur=0x8 adr=0x3 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=72 acr=0x6c ipr=0x6 inr=0xa300004 bur=0x8 adr=0x3 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hel' <- 'l'
DEBUG:root:tick=75 acr=0x6c ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0x3
DEBUG:root:inc *spr
DEBUG:root:tick=79 acr=0x6c ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x4 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=81 acr=0x6c ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x4 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:ld **spr
DEBUG:root:tick=86 acr=0x6f ipr=0x4 inr=0x2500000 bur=0x8 adr=0x4 dar=0x6f spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:cmp #0x0
DEBUG:root:tick=88 acr=0x6f ipr=0x5 inr=0x9100000 bur=0x8 adr=0x4 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=89 acr=0x6f ipr=0x6 inr=0xa300004 bur=0x8 adr=0x4 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hell' <- 'o'
DEBUG:root:tick=92 acr=0x6f ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x6f spr=0x1ffc flr=0x1 stack_top=0x4
DEBUG:root:inc *spr
DEBUG:root:tick=96 acr=0x6f ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x5 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=98 acr=0x6f ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x5 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:ld **spr
DEBUG:root:tick=103 acr=0x2c ipr=0x4 inr=0x2500000 bur=0x8 adr=0x5 dar=0x2c spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:cmp #0x0
DEBUG:root:tick=105 acr=0x2c ipr=0x5 inr=0x9100000 bur=0x8 adr=0x5 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=106 acr=0x2c ipr=0x6 inr=0xa300004 bur=0x8 adr=0x5 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello' <- ','
DEBUG:root:tick=109 acr=0x2c ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x2c spr=0x1ffc flr=0x1 stack_top=0x5
DEBUG:root:inc *spr
DEBUG:root:tick=113 acr=0x2c ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x6 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=115 acr=0x2c ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x6 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:ld **spr
DEBUG:root:tick=120 acr=0x20 ipr=0x4 inr=0x2500000 bur=0x8 adr=0x6 dar=0x20 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:cmp #0x0
DEBUG:root:tick=122 acr=0x20 ipr=0x5 inr=0x9100000 bur=0x8 adr=0x6 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=123 acr=0x20 ipr=0x6 inr=0xa300004 bur=0x8 adr=0x6 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello,' <- ' '
DEBUG:root:tick=126 acr=0x20 ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x20 spr=0x1ffc flr=0x1 stack_top=0x6
DEBUG:root:inc *spr
DEBUG:root:tick=130 acr=0x20 ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x7 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=132 acr=0x20 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x7 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:ld **spr
DEBUG:root:tick=137 acr=0x77 ipr=0x4 inr=0x2500000 bur=0x8 adr=0x7 dar=0x77 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:cmp #0x0
DEBUG:root:tick=139 acr=0x77 ipr=0x5 inr=0x9100000 bur=0x8 adr=0x7 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=140 acr=0x77 ipr=0x6 inr=0xa300004 bur=0x8 adr=0x7 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, ' <- 'w'
DEBUG:root:tick=143 acr=0x77 ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x77 spr=0x1ffc flr=0x1 stack_top=0x7
DEBUG:root:inc *spr
DEBUG:root:tick=147 acr=0x77 ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x8 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=149 acr=0x77 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x8 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:ld **spr
DEBUG:root:tick=154 acr=0x6f ipr=0x4 inr=0x2500000 bur=0x8 adr=0x8 dar=0x6f spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:cmp #0x0
DEBUG:root:tick=156 acr=0x6f ipr=0x5 inr=0x9100000 bur=0x8 adr=0x8 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=157 acr=0x6f ipr=0x6 inr=0xa300004 bur=0x8 adr=0x8 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, w' <- 'o'
DEBUG:root:tick=160 acr=0x6f ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x6f spr=0x1ffc flr=0x1 stack_top=0x8
DEBUG:root:inc *spr
DEBUG:root:tick=164 acr=0x6f ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0x9 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=166 acr=0x6f ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0x9 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:ld **spr
DEBUG:root:tick=171 acr=0x72 ipr=0x4 inr=0x2500000 bur=0x8 adr=0x9 dar=0x72 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:cmp #0x0
DEBUG:root:tick=173 acr=0x72 ipr=0x5 inr=0x9100000 bur=0x8 adr=0x9 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=174 acr=0x72 ipr=0x6 inr=0xa300004 bur=0x8 adr=0x9 dar=0x0 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, wo' <- 'r'
DEBUG:root:tick=177 acr=0x72 ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x72 spr=0x1ffc flr=0x1 stack_top=0x9
DEBUG:root:inc *spr
DEBUG:root:tick=181 acr=0x72 ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0xa spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=183 acr=0x72 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0xa spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:ld **spr
DEBUG:root:tick=188 acr=0x6c ipr=0x4 inr=0x2500000 bur=0x8 adr=0xa dar=0x6c spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:cmp #0x0
DEBUG:root:tick=190 acr=0x6c ipr=0x5 inr=0x9100000 bur=0x8 adr=0xa dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=191 acr=0x6c ipr=0x6 inr=0xa300004 bur=0x8 adr=0xa dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, wor' <- 'l'
DEBUG:root:tick=194 acr=0x6c ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x6c spr=0x1ffc flr=0x1 stack_top=0xa
DEBUG:root:inc *spr
DEBUG:root:tick=198 acr=0x6c ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0xb spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=200 acr=0x6c ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0xb spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:ld **spr
DEBUG:root:tick=205 acr=0x64 ipr=0x4 inr=0x2500000 bur=0x8 adr=0xb dar=0x64 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:cmp #0x0
DEBUG:root:tick=207 acr=0x64 ipr=0x5 inr=0x9100000 bur=0x8 adr=0xb dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=208 acr=0x64 ipr=0x6 inr=0xa300004 bur=0x8 adr=0xb dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, worl' <- 'd'
DEBUG:root:tick=211 acr=0x64 ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x64 spr=0x1ffc flr=0x1 stack_top=0xb
DEBUG:root:inc *spr
DEBUG:root:tick=215 acr=0x64 ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0xc spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=217 acr=0x64 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0xc spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:ld **spr
DEBUG:root:tick=222 acr=0x21 ipr=0x4 inr=0x2500000 bur=0x8 adr=0xc dar=0x21 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:cmp #0x0
DEBUG:root:tick=224 acr=0x21 ipr=0x5 inr=0x9100000 bur=0x8 adr=0xc dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=225 acr=0x21 ipr=0x6 inr=0xa300004 bur=0x8 adr=0xc dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:st *0x15b4
DEBUG:root:output: 'Hello, world' <- '!'
DEBUG:root:tick=228 acr=0x21 ipr=0x7 inr=0x32015b4 bur=0x8 adr=0x15b4 dar=0x21 spr=0x1ffc flr=0x1 stack_top=0xc
DEBUG:root:inc *spr
DEBUG:root:tick=232 acr=0x21 ipr=0x8 inr=0xd400000 bur=0x8 adr=0x1ffc dar=0xd spr=0x1ffc flr=0x1 stack_top=0xd
DEBUG:root:br *ipr+0xffffb
DEBUG:root:tick=234 acr=0x21 ipr=0x3 inr=0xc3ffffb bur=0x8 adr=0x1ffc dar=0xd spr=0x1ffc flr=0x1 stack_top=0xd
DEBUG:root:ld **spr
DEBUG:root:tick=239 acr=0x0 ipr=0x4 inr=0x2500000 bur=0x8 adr=0xd dar=0x0 spr=0x1ffc flr=0x1 stack_top=0xd
DEBUG:root:cmp #0x0
DEBUG:root:tick=241 acr=0x0 ipr=0x5 inr=0x9100000 bur=0x8 adr=0xd dar=0x0 spr=0x1ffc flr=0x5 stack_top=0xd
DEBUG:root:brz *ipr+0x4
DEBUG:root:tick=243 acr=0x0 ipr=0x9 inr=0xa300004 bur=0x5 adr=0xd dar=0x0 spr=0x1ffc flr=0x5 stack_top=0xd
DEBUG:root:pop
DEBUG:root:tick=247 acr=0xd ipr=0xa inr=0x7000000 bur=0x5 adr=0x1ffc dar=0xd spr=0x1ffd flr=0x5 stack_top=0x0
DEBUG:root:sub *spr
DEBUG:root:tick=250 acr=0xd ipr=0xb inr=0x11400000 bur=0x5 adr=0x1ffd dar=0x0 spr=0x1ffd flr=0x5 stack_top=0x0
DEBUG:root:popn
DEBUG:root:tick=251 acr=0xd ipr=0xc inr=0x8000000 bur=0x5 adr=0x1ffd dar=0x0 spr=0x1ffe flr=0x5 stack_top=0xf
DEBUG:root:ret
DEBUG:root:tick=255 acr=0xd ipr=0xf inr=0x5000000 bur=0x5 adr=0x1ffe dar=0xf spr=0x1fff flr=0x5 stack_top=?
DEBUG:root:halt
Hello, world!
instr:  90 ticks:  255
```

Пример проверки исходного кода:

```bash
% poetry run pytest . -v
============================================================================================= test session starts ==============================================================================================
platform darwin -- Python 3.12.0, pytest-7.4.3, pluggy-1.3.0 -- /Users/andryssssss/Library/Caches/pypoetry/virtualenvs/lispfuck-qe1KGZJD-py3.12/bin/python
cachedir: .pytest_cache
rootdir: /Users/andryssssss/Programming/Projects/Python/ITMO/ComputerArchitectureLab3
configfile: pyproject.toml
plugins: golden-0.2.2
collected 7 items                                                                                                                                                                                              

integration_test.py::test_translator_and_machine[golden/func.yml] PASSED                                                                                                                                 [ 14%]
integration_test.py::test_translator_and_machine[golden/if.yml] PASSED                                                                                                                                   [ 28%]
integration_test.py::test_translator_and_machine[golden/math.yml] PASSED                                                                                                                                 [ 42%]
integration_test.py::test_translator_and_machine[golden/hello.yml] PASSED                                                                                                                                [ 57%]
integration_test.py::test_translator_and_machine[golden/prob5.yml] PASSED                                                                                                                                [ 71%]
integration_test.py::test_translator_and_machine[golden/hello_user_name.yml] PASSED                                                                                                                      [ 85%]
integration_test.py::test_translator_and_machine[golden/echo.yml] PASSED                                                                                                                                 [100%]

============================================================================================== 7 passed in 0.61s ===============================================================================================
% poetry run ruff check .
% poetry run ruff format .
5 files left unchanged
```

Для статистики:

```text
| ФИО                            | алг   | LoC | code байт | code инстр. | инстр. | такт. | вариант                                                                        |
| Кривошеев Андрей Александрович | hello | 2   | 186       | 16          | 90     | 255   | `lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob5 | 8bit` |
| Кривошеев Андрей Александрович | echo  | 2   | 158       | 37          | 106    | 299   | `lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob5 | 8bit` |
| Кривошеев Андрей Александрович | prob5 | 16  | 670       | 125         | 2666   | 6980  | `lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob5 | 8bit` |
```
