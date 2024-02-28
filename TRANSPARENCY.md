# Transparency FAQ for BitBLAS

## What is BitBLAS?

BitBLAS is a lightweight framework designed for generating high-performance CUDA/HIP code for BLAS (Basic Linear Algebra Subprograms) operators, emphasizing swizzling and layout propagation. It leverages a Domain-Specific Language (DSL), specifically TIR Script, to offer flexibility and efficiency in mathematical computations. BitBLAS aims to provide performance comparable to cuBLAS while introducing more flexibility and efficiency through its unique features.

## What can BitBLAS do?

BitBLAS enhances the performance and flexibility of linear algebra computations with features like:

- Auto Tensorization: Automatically optimizes code for various data types and operators, supporting FP16, INT8, and mixed precision operations.
- Dynamic Symbolic Support: Facilitates kernel generation with dynamic shapes, enabling efficient computation for variable data sizes.
- High-Performance Computing: Offers optimized performance for different data operations, including FP16xFP16, FP16xINT4/2/1, INT8xINT8, and INT8xINT4/2/1, among others.

## What are BitBLAS's intended uses?

BitBLAS is intended for developers and researchers who require high-performance linear algebra computations in their CUDA/HIP-based applications. It is particularly beneficial for:

- Machine Learning and Deep Learning: Accelerating training and inference computations.
- Scientific Computing: Handling large-scale linear algebra operations efficiently.
- High-Performance Computing (HPC): Enhancing performance in computationally intensive applications.

## Data Handling and Privacy

This project is committed to protecting privacy and ensuring a secure environment for all users. It is designed with the following principles in mind:

- No User Data Collection: The project does not collect, process, or store any personal or privacy-sensitive data from users. Users can utilize the project's features without the concern of their data being recorded or misused.

- Transparency: We believe in complete transparency with our community. As such, we clearly state that no user data is collected or processed at any stage of the project's usage.

- User Control and Privacy: Since the project does not involve user data, individuals retain full control over their information. Users can interact with the project knowing their privacy is safeguarded.

## Security Considerations

The security of the project and its users is paramount. Despite not handling user data, we adhere to best practices in software development to ensure the project's integrity and safety:

- Regular Security Audits: The project undergoes regular security reviews and audits to identify and remediate any potential vulnerabilities, ensuring the highest level of security.

- Open Source Security: As an open-source project, our code is available for review, allowing the community to examine and contribute to the project's security.

- Security Incident Response: In the unlikely event of a security issue, we have established procedures for prompt and effective response to investigate and address the concern.

- Community Involvement: We encourage the community to report any security concerns or suggestions for improvement. Our project's success and security are enhanced by active community participation and feedback.

## Compliance and Licensing

As a project initiated and released by Microsoft, we adhere strictly to legal and regulatory standards to ensure our contributions meet the highest compliance criteria. Here are key points regarding our compliance and licensing practices:

- Microsoft's Commitment: This project is part of Microsoft's commitment to supporting and contributing to the open-source community. We ensure that all contributions are compliant with current legal standards and practices.

- MIT License: The project is licensed under the MIT License, one of the most permissive and open licenses available. This license allows for almost unrestricted freedom to use, modify, and distribute the project, providing that the license and copyright notice are included with any substantial portion of the software.

- License Clarity: We have clearly indicated the licensing terms within the project repository to ensure that all users and contributors are informed of their rights and obligations when using or contributing to the project.

