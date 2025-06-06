"""Implements the Cypher class, the base class for stream cyphers."""

import os
import queue
import stat
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Callable, Literal

__all__: list[str] = []


class _Cypher(ABC):
    """Abstract base class for cyphers; provides utils that are common to all subclasses."""

    @abstractmethod
    def _generate_delimiter(self, seed: bytes) -> tuple[memoryview, bytes]:
        """Create a delimiter sequence using the key stream and update the seed.

        Args:
            seed (bytes): Seed used for generating the delimiter.

        Returns:
            tuple[memoryview, bytes]: The delimiter and the refreshed seed.
        """
        pass

    @abstractmethod
    def _hash(
        self,
        key: bytes | bytearray | memoryview,
        msg_list: list[bytes | bytearray | memoryview],
        use_hmac: bool = False,
    ) -> bytes:
        """Generate a Keyed Hash or Hash-based Message Authentication Code (HMAC).

        Each element in `msg_list` is sequentially fed into the Hash as message data.

        Args:
            key (bytes or bytearray or memoryview): The key for the keyed hash or HMAC.
            msg_list (list of bytes or bytearray or memoryview): List of message parts to hash with the key.
            use_hmac (bool): If True, the key is used for HMAC; otherwise, it's a keyed hash. Defaults to False.

        Returns:
            bytes: The resulting hash digest.
        """
        pass

    @abstractmethod
    def encode(
        self, message: bytes | bytearray | memoryview, seed: bytes
    ) -> tuple[bytearray, bytes]:
        """Encrypt a message.

        Args:
            message (bytes or bytearray or memoryview): Message to encode.
            seed (bytes): Initial seed for encryption.

        Returns:
            tuple[bytearray, bytes]: Encrypted message and final seed.

        Raises:
            ValueError: If the delimiter appears in the message.
        """
        pass

    @abstractmethod
    def decode(
        self, cyphertext: bytes | bytearray | memoryview, seed: bytes
    ) -> tuple[bytearray, bytes]:
        """Decrypt an encoded message.

        Args:
            cyphertext (bytes or bytearray or memoryview): Encrypted and obfuscated message.
            seed (bytes): Initial seed for decryption.

        Returns:
            tuple[bytearray, bytes]: Decrypted message and final seed.

        Raises:
            ValueError: If the authentication tag does not match.
        """
        pass

    def process_file(
        self,
        mode: Literal["encode", "decode"],
        input_file: str | Path | IO[bytes],
        output_file: str | Path | IO[bytes],
        seed: bytes,
        buffer_size: int = 1024 * 1024,
        read_queue_size: int = 4,
        write_queue_size: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Processes a file or stream in blocks using the provided Cypher for encryption or decryption.

        Args:
            mode (Literal["encode", "decode"]): Operation mode ("encode" for encryption, "decode" for decryption).
            input_file (str or Path or IO[bytes]): Path or file-like object for input.
            output_file (str or Path or IO[bytes]): Path or file-like object for output.
            seed (bytes): Initial seed for processing.
            buffer_size (int): Bytes to read at a time. Defaults to `1024 * 1024` (1MB).
            read_queue_size (int): Maximum number of data blocks buffered in the
                queue between the IO reader thread and the main processing thread. Defaults to 4.
            write_queue_size (int): Maximum number of data blocks buffered in the
                queue between the main processing thread and the IO writer thread. Defaults to 4.
            progress_callback (Callable, optional): Callback for progress reporting.
                Receives two arguments: `bytes_processed` and `total_size`. Defaults to None.

        Raises:
            ValueError: If `mode` is not "encode" or "decode".
            TypeError: If `buffer_size`, `read_queue_size`, or `write_queue_size` is not an integer.
            ValueError: If `buffer_size`, `read_queue_size`, or `write_queue_size` is not a positive integer.
            exception: If an unexpected error occurs in the reader or writer threads.
        """
        # Input validation
        if mode not in ("encode", "decode"):
            raise ValueError("Invalid mode. Use 'encode' or 'decode'.")
        if not isinstance(buffer_size, int):
            raise TypeError("buffer_size must be an integer.")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer.")
        if not isinstance(read_queue_size, int):
            raise TypeError("read_queue_size must be an integer.")
        if read_queue_size <= 0:
            raise ValueError("read_queue_size must be a positive integer.")
        if not isinstance(write_queue_size, int):
            raise TypeError("write_queue_size must be an integer.")
        if write_queue_size <= 0:
            raise ValueError("write_queue_size must be a positive integer.")

        # Open the input and output if necessary
        def _open_if_path(obj: str | Path | IO[bytes], mode: str) -> IO[bytes]:
            if isinstance(obj, str):
                return open(obj, mode)
            elif isinstance(obj, Path):
                return obj.open(mode)
            else:
                return obj

        infile = _open_if_path(input_file, "rb")
        outfile = _open_if_path(output_file, "wb")

        # Progress tracking setup
        total_size = 0
        bytes_processed = 0
        try:
            if hasattr(infile, "fileno"):
                fileno = infile.fileno()
                if not os.isatty(fileno):
                    file_stat = os.fstat(fileno)
                    if stat.S_ISREG(file_stat.st_mode):
                        total_size = file_stat.st_size
        except Exception:
            pass  # Not a regular file or can't determine size

        if total_size <= 0:
            progress_callback = None
        elif progress_callback is not None:
            progress_callback(0, total_size)

        # Reader and Writer threads used for asynchronous IO
        read_q: queue.Queue[bytes | bytearray | memoryview] = queue.Queue(maxsize=read_queue_size)
        write_q: queue.Queue[bytes | bytearray | memoryview] = queue.Queue(maxsize=write_queue_size)
        exception_queue: queue.Queue[BaseException] = queue.Queue()

        def queue_get(
            q: queue.Queue[bytes | bytearray | memoryview],
        ) -> bytes | bytearray | memoryview:
            # Gets from queue in a blocking way with timeout, as long as the exception queue is empty
            # Returns the data if successful, Empty data if an error occurs
            while exception_queue.empty():
                try:
                    return q.get(block=True, timeout=0.05)
                except queue.Empty:
                    continue
            return b""

        def queue_put(
            q: queue.Queue[bytes | bytearray | memoryview], data: bytes | bytearray | memoryview
        ) -> bool:
            # Puts in queue in a blocking way with timeout, as long as the exception queue is empty
            # Returns True if successful, False if an error occurs
            while exception_queue.empty():
                try:
                    q.put(data, block=True, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def reader_thread_func() -> None:
            try:
                while exception_queue.empty():
                    block = infile.read(buffer_size)
                    if not queue_put(read_q, block):
                        break  # Exception occurred
                    if not block:  # Empty block indicates EOF
                        break
            except Exception as e:
                exception_queue.put(e)

        def writer_thread_func() -> None:
            try:
                while exception_queue.empty():
                    data = queue_get(write_q)
                    if not data:  # Signal to stop or exception occurred
                        break
                    outfile.write(data)
            except Exception as e:
                exception_queue.put(e)

        reader_thread = threading.Thread(target=reader_thread_func, daemon=True)
        writer_thread = threading.Thread(target=writer_thread_func, daemon=True)
        reader_thread.start()
        writer_thread.start()

        try:
            # Generate the initial block delimiter
            current_seed = self._hash(seed, [b"block_delimiter"])
            block_delimiter, current_seed = self._generate_delimiter(current_seed)
            delimiter_size = len(block_delimiter)

            if mode == "encode":
                first_block = True
                while exception_queue.empty():
                    # Read from the file
                    block = queue_get(read_q)
                    if not block:
                        break  # End of file or exception occurred

                    if not first_block:
                        # Write a fixed delimiter to mark the end of the previous block
                        if not queue_put(write_q, block_delimiter):
                            break
                        # Refresh the block delimiter
                        block_delimiter, current_seed = self._generate_delimiter(current_seed)
                    else:
                        first_block = False

                    # Encode the content block
                    processed_block, current_seed = self.encode(block, current_seed)

                    # Write the processed block to the output file
                    if not queue_put(write_q, processed_block):
                        break

                    if progress_callback:
                        bytes_processed += len(block)
                        progress_callback(bytes_processed, total_size)
            elif mode == "decode":
                last_block = False
                buffer = bytearray()
                while exception_queue.empty():
                    block = queue_get(read_q)

                    buffer.extend(block)
                    while exception_queue.empty():
                        delim_index = buffer.find(block_delimiter)
                        if delim_index == -1:
                            # No delimiter found
                            if not block and buffer:
                                # This is EOF, set delim_index to the end of the buffer
                                delim_index = len(buffer)
                                last_block = True
                            else:
                                break  # No complete block in buffer yet or EOF handled

                        # Extract the complete block up to the delimiter
                        complete_block = memoryview(buffer)[:delim_index]

                        # Decode the complete block
                        processed_block, current_seed = self.decode(complete_block, current_seed)
                        if not queue_put(write_q, processed_block):
                            break

                        if last_block:
                            buffer = bytearray()  # Clear buffer
                            break

                        # Remove the processed block and delimiter from the buffer
                        buffer = buffer[delim_index + delimiter_size :]

                        # Refresh block delimiter
                        block_delimiter, current_seed = self._generate_delimiter(current_seed)

                    if progress_callback:
                        bytes_processed += len(block)
                        progress_callback(bytes_processed, total_size)

                    if not block and not buffer:  # End of file with nothing left to process or exception occurred
                        break
            else:
                raise ValueError("Invalid mode. Use 'encode' or 'decode'.")
        except BaseException as main_exc:
            # Signal the threads to stop
            exception_queue.put(main_exc)
            raise
        finally:
            # Wait for threads to finish
            if reader_thread.is_alive():
                reader_thread.join()
            if writer_thread.is_alive():
                try:
                    # Signal the writer thread to stop
                    queue_put(write_q, b"")
                except Exception:
                    # Ensure that the files are closed below
                    pass
                writer_thread.join()

            # Close the input and output files if they were opened
            if isinstance(input_file, (str, Path)):
                infile.close()
            if isinstance(output_file, (str, Path)):
                outfile.close()

            # Check for exceptions from the threads
            if not exception_queue.empty():
                exception = exception_queue.get()
                raise exception
