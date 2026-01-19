> [!TIP]
> 提高开发效率以及避免踩坑的一些建议:
> 关闭 AI 补全（显著提高降低效率，但是能够了解到更多的细节处理）
> 使用 Python 类型标注系统，并将 Pylance 类型检查设置为标准，这样能在静态检查出绝大多数类型、参数不匹配的问题
> 实现 BPE 前首先厘清其中各个数据结构和流程的概念，例如语料库、pre-token、token、预分词，然后再动手
> 使用 logger 和 tqdm 随时随地打印进度，以便对各个组件耗时和瓶颈组件有个直观的认识

# BPE 分词器
分词器的整个构建流程:
> [词表初始化]
> 构造初始词表，包括 256 个 ASCII 字符和 special tokens
> [pre-tokenize]
> 将给定语料按照给定正则表达式划分为 pre-token, 并统计 pre-token 各个词频
## Pre-Tokenizer：
Pre-Tokenizer 类的接口如下所示, 其对外提供一个 pre_tokenize 方法为给定字节流生成 pre-token 迭代器, 一个 __call__ 方法将给定语料库文件转换为 pre-token 出现频次的字典
```Python
class PreTokenizer(ABC):
    @staticmethod
    def _merge_pre_token_counts(*pre_token_counts: PreTokenCount) -> PreTokenCount:
        """Merge multiple PreTokenCount dictionaries into one.

        Returns:
            PreTokenCount: The merged PreTokenCount.
        """

    def _process_chunk(self, chunk: Chunk, special_tokens: list[Token]) -> PreTokenCount:
        """
        Process a single chunk of text and return the pre-token counts.

        Args:
            chunk (Chunk): The chunk of text to process.

        Returns:
            PreTokenCount: A dictionary-like object mapping pre-tokens to their counts.
        """

    def pre_tokenize(self, str_bytes: bytes, special_token_list: list[Token]) -> Iterator[Token]:
        """
        Pre-tokenize the given bytes string.

        Args:
            str_bytes (bytes): The input bytes string to pre-tokenize.
            special_token_list (list[Token]): The list of special tokens.
        Returns:
            Iterator[Token]: An iterator over the pre-tokens.
        """

    @abstractmethod
    def __call__(self, corpos_path: str, split_special_token: Token, special_tokens: list[Token]) -> PreTokenCount:
        """
        Pre-tokenize the given corpus.

        Args:
            corpos_path (str): Path to the corpus file.
            split_special_token (token): The special token used to split the corpus.
            special_tokens (list[Token]): List of special tokens.

        Returns:
            PreTokenCount: A dictionary-like object mapping pre-tokens to their counts.
        """
```
> [!TIP]
> 在具体实现时，为了采用多进程以提高处理效率， _process_chunk 负责统计一个 chunk 中 pre-token 出现的频次
> 如下所示，一个语料库会被以 special token 为边界划分为多个 chunk，每个 chunk 由多篇被 special token 分隔的文档组成
![_process_chunk](assets/_process_chunk.png)
> [计算 BPE merges]
> 基于给定的 pre-token 的词频计算结果运行BPE算法, 合并出现频次最高相邻 pre-token 对作为一个 token, 并产出 merge pair 和 词表, 不断重复这个过程直至词表达到目标

> [!IMPORTANT]
> 在 _process_chunk 的帮助下，多进程的实现就变得很简单, 每个进程每次处理一个 chunk
> 只需将给定的语料库划分为指定数量的 chunk，然后使用进程池派发这些任务
> _process_chunk 先按照 special token 将 chunk 划分为 document
> 然后使用 Counter 模块和给定的 pre-token 的正则表达式统计单篇 document 中每个 pre-token 的出现次数
> 统计每个 chunk 中各个 pre-token 的出现次数
> 最后将每个 chunk 的词频归约在一起即可
```Python
class MultiProcessPreTokenizer(PreTokenizer):
    def _process_chunk_with_boundry(
        self, corpos_path: str, start: int, end: int, special_tokens: list[Token]
    ) -> PreTokenCount:
        with open(corpos_path, mode="br") as f:
            f.seek(start)
            chunk = f.read(end - start)
            pre_token_count = self._process_chunk(chunk, special_tokens)
        return pre_token_count

    def __call__(self, corpos_path: str, split_special_token: Token, special_tokens: list[Token]) -> PreTokenCount:
        final_pre_token_count: PreTokenCount = defaultdict(int)

        start_time = time.time()
        file_size = os.path.getsize(corpos_path)
        num_cpus = cpu_count()

        desired_chunks = num_cpus * 100

        chunk_boundaries = find_chunk_boundaries(
            file_path=corpos_path,
            desired_num_chunks=desired_chunks,
            split_special_token=split_special_token,
        )

        chunks_args = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunks_args.append((corpos_path, start, end, special_tokens))

        logger.info(f"Splitting task into {len(chunks_args)} chunks.")

        with Pool(processes=num_cpus * 2) as pool:
            chunk_iter = pool.imap_unordered(self._worker_wrapper, chunks_args)

            for chunk_result in tqdm.tqdm(chunk_iter, total=len(chunks_args), desc="Pre-tokenizing"):
                for token, count in chunk_result.items():
                    final_pre_token_count[token] += count

        end_time = time.time()
        logger.info(
            "Takes {:.2f} seconds to pre-tokenize, speed: {:.2f} bytes/second",
            end_time - start_time,
            file_size / (end_time - start_time),
        )
        return final_pre_token_count

    @staticmethod
    def _worker_wrapper(args):
        tokenizer_instance = MultiProcessPreTokenizer()
        return tokenizer_instance._process_chunk_with_boundry(*args)
```
## TokenizerTrainer：
TokenizerTrainer 负责在给定词频统计结果上运行 BPE 算法. 找出出现频次最高的组合并合并, 为了高效实现 BEP 算法, 首先要回答以下几个问题:
1. 怎么初始化各个组合（Pair）的出现频次？
  - Pre-Tokenizer 提供的是每个 pre-token 的频次，可以通过遍历每个 pre-token 中可能出现的组合并累加频次即可。例如，对于 pre-token Hello ，可以贡献 (H, e), (e, l), (l, l), (l, o), (o, 空格) 这六组频次。
2. 合并后怎么维护各个组合的频次？
    每当一个组合被合并后，各个组合的频次需要更新，以 Hello 中 el 合并为例，需要更新的频次包括：

    新产生组合： el 作为一个独立的 token，与其邻接的两个 token 将产生新的组合 (H, el) 和 (el, l)，这两个组合需要新增。
    频次减少的组合：el 合并之后，这个组合不复存在，其词频需要设置为 0。除此之外，与被合并的 token 邻接的其它 token 组成的组合频次也许相应减少，即 (H, e) 和 (l, l) 的频次需要减少 Hello 这个单词的次数。
3. 合并后如何快速定位到受影响的 pre-token？
    在上一问中，我们解决了在 pre-token 已知的情况下词频的维护逻辑。但是如何快速找到受影响的 pre-token？朴素的方法是直接遍历整个 pre-token，显然每次都要遍历的方案完全不可接受。可行的方案是我们维护一张 pair 到 pre-token 的映射表，表示含有这个 pair 的 pre-token 列表。这张表在组合频次初始化时也一起被初始化，在组合被合并时也一起被更新，从而在 merge 过程中找到受影响的 pre-token 列表。

4. 如何记录当前 pre-token 的状态？
    pre-teken 的状态指的是当前的 pre-token 由哪些 token 组成。举个栗子，BPE 算法刚开始时，Hello 这个 pre-token 的是由 (H, e, l, l, o, 空格) 这六个 token 组成的，在算法后期，其可能是由 (Hel, l, o空格) 这三个 token 组成。此时如果需要合并 (l, o空格)，在更新组合的频次时就要知道 l 的前一个 token 是啥，而非简单查询 l 前一个字符是什么，因此我们还需要一个字典来维护每个 pre-token 当前的状态。

5. 如何获取频率最大的组合？
    朴素的方案是每次都遍历整个词频表，时间复杂度是 O(n)。我们可以使用最大堆来优化这一过程，从而可以将单次获取并维护最大值的时间复杂度降为 O(log n)

