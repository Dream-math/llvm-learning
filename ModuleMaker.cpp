//===- examples/ModuleMaker/ModuleMaker.cpp - Example project ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This programs is a simple example that creates an LLVM module "from scratch",
// emitting it as a bitcode file to standard out.  This is just to show how
// LLVM projects work and to demonstrate some of the LLVM APIs.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/SourceMgr.h>

#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/IntrinsicsARM.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/IntrinsicsX86.h>

#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>


#include <llvm/Support/Alignment.h>

#include <llvm/CodeGen/TargetLoweringObjectFileImpl.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

using namespace llvm;

int main() {
  LLVMContext Context;

  // Create the "module" or "program" or "translation unit" to hold the
  // function
  Module *M = new Module("test", Context);

  // Create the main function: first create the type 'int ()'
  FunctionType *FT =
    FunctionType::get(Type::getInt32Ty(Context), /*not vararg*/false);

  // By passing a module as the last parameter to the Function constructor,
  // it automatically gets appended to the Module.
  Function *F = Function::Create(FT, Function::ExternalLinkage, "main", M);

  // Add a basic block to the function... again, it automatically inserts
  // because of the last argument.
  BasicBlock *BB = BasicBlock::Create(Context, "EntryBlock", F);

  // Get pointers to the constant integers...
  Value *Two = ConstantInt::get(Type::getInt32Ty(Context), 2);
  Value *Three = ConstantInt::get(Type::getInt32Ty(Context), 3);

  // Create the add instruction... does not insert...
  Instruction *Add = BinaryOperator::Create(Instruction::Add, Two, Three,
                                            "addresult");

  // explicitly insert it into the basic block...
  BB->getInstList().push_back(Add);

  // Create the return instruction and add it to the basic block
  BB->getInstList().push_back(ReturnInst::Create(Context, Add));

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  const std::string target_triple = "riscv64-unknown-linux-gnu";
  const std::string mcpu = "generic-rv64";
  const std::string mattr = "+64bit,+a,+c,+d,+f,+m,+experimental-v";
  std::string err;

  llvm::TargetOptions opt = llvm::TargetOptions();
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  // for double float 
  opt.FloatABIType = llvm::FloatABI::Hard;

  const llvm::Target* llvm_target = llvm::TargetRegistry::lookupTarget(target_triple, err);
  if (!llvm_target) throw std::runtime_error("Couldn't find target.");
  llvm::TargetMachine* tm =
    llvm_target->createTargetMachine(target_triple, mcpu, mattr, opt, llvm::Reloc::PIC_);

  std::error_code ecode;
  // std::string file_name = "D1/dll_deploy/lib/riscv.o";
  std::string file_name = "D1/dll_deploy/lib/riscv.s";
  llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::F_None);
  llvm::legacy::PassManager pass;

  // tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_ObjectFile);
  tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_AssemblyFile);
  pass.run(*M);
  // dest.close();

  // Output the bitcode file to stdout
  // WriteBitcodeToFile(*M, outs());

  // Delete the module and all of its contents.
  delete M;
  return 0;
}
